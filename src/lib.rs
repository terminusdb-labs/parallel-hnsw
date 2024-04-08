pub mod bigvec;
pub mod pq;
mod priority_queue;
mod search;
pub mod serialize;
mod types;

pub use serialize::SerializationError;
pub use types::*;

use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    path::Path,
    slice::Iter,
    sync::{
        atomic::{self, AtomicUsize},
        RwLock,
    },
};

use itertools::Itertools;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{Distribution, Exp};
use rayon::prelude::*;
use std::fmt::Debug;

use crate::priority_queue::PriorityQueue;

pub enum WrappedBorrowable<'a, T: ?Sized, Borrowable: Deref<Target = T> + 'a> {
    Left(Borrowable),
    Right(&'a T),
}

impl<'a, T: ?Sized, Borrowable: Deref<Target = T> + 'a> Deref
    for WrappedBorrowable<'a, T, Borrowable>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            WrappedBorrowable::Left(b) => b,
            WrappedBorrowable::Right(b) => b,
        }
    }
}

pub trait Comparator: Sync + Clone {
    type T;
    type Borrowable<'a>: Deref<Target = Self::T>
    where
        Self: 'a;
    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_>;
    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32;
    fn lookup_abstract<'a: 'b, 'b>(
        &'a self,
        v: AbstractVector<'b, Self::T>,
    ) -> WrappedBorrowable<'b, Self::T, Self::Borrowable<'b>> {
        match v {
            AbstractVector::Stored(i) => WrappedBorrowable::Left(self.lookup(i)),
            AbstractVector::Unstored(v) => WrappedBorrowable::Right(v),
        }
    }
    fn compare_vec(&self, v1: AbstractVector<Self::T>, v2: AbstractVector<Self::T>) -> f32 {
        let v1 = self.lookup_abstract(v1);
        let v2 = self.lookup_abstract(v2);
        self.compare_raw(&*v1, &*v2)
    }
}

pub trait Serializable: Sized {
    type Params;
    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError>;
    fn deserialize<P: AsRef<Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, SerializationError>;
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Layer<C> {
    pub comparator: C,
    pub neighborhood_size: usize,
    pub nodes: Vec<VectorId>,
    pub neighbors: Vec<NodeId>,
}

impl<C: Comparator> Clone for Layer<C> {
    fn clone(&self) -> Self {
        Self {
            comparator: self.comparator.clone(),
            neighborhood_size: self.neighborhood_size,
            nodes: self.nodes.clone(),
            neighbors: self.neighbors.clone(),
        }
    }
}

unsafe impl<C: Comparator> Sync for Layer<C> {}

impl<C: Comparator> AsRef<Layer<C>> for Layer<C> {
    fn as_ref(&self) -> &Layer<C> {
        self
    }
}

type NodeDistances = Vec<(NodeId, f32)>;

impl<C> Layer<C> {
    #[allow(unused)]
    pub fn get_node(&self, v: VectorId) -> Option<NodeId> {
        self.nodes.binary_search(&v).ok().map(NodeId)
    }

    pub fn get_vector(&self, n: NodeId) -> VectorId {
        if n.0 > self.nodes.len() {
            eprintln!("nodes: {:?}", self.nodes);
            eprintln!("neighborhood: {:?}", self.neighbors);
        }
        self.nodes[n.0]
    }

    pub fn get_neighbors(&self, n: NodeId) -> &[NodeId] {
        &self.neighbors[(n.0 * self.neighborhood_size)..((n.0 + 1) * self.neighborhood_size)]
    }
    pub fn get_neighbors_mut(&mut self, n: NodeId) -> &mut [NodeId] {
        &mut self.neighbors[(n.0 * self.neighborhood_size)..((n.0 + 1) * self.neighborhood_size)]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl<C: Comparator> Layer<C> {
    pub fn nearest_neighbors(
        &self,
        n: NodeId,
        number_of_nodes: usize,
        probe_depth: usize,
    ) -> Vec<(NodeId, f32)> {
        let v = self.get_vector(n);
        let mut candidates = PriorityQueue::new(number_of_nodes);
        candidates.insert(n, f32::MAX);
        self.closest_nodes(AbstractVector::Stored(v), &mut candidates, probe_depth);
        candidates.iter().collect()
    }

    pub fn closest_nodes(
        &self,
        v: AbstractVector<C::T>,
        candidates: &mut PriorityQueue<NodeId>,
        mut probe_depth: usize,
    ) -> usize {
        assert!(!candidates.is_empty());
        let mut visit_queue: Vec<(NodeId, f32, NodeDistance)> = candidates
            .iter()
            .map(|(n, f)| (n, f, NodeDistance::ZERO))
            .collect();
        visit_queue.reverse();
        let mut visited: HashSet<NodeId> = candidates.iter().map(|(n, _)| n).collect();
        //eprintln!("------------------------------------");
        //eprintln!("Initial visit queue: {visit_queue:?}");
        let mut highest_improvement = 0;
        while let Some((next, _, node_distance)) = visit_queue.pop() {
            //eprintln!("...");
            //eprintln!("working with next: {next:?}");
            //visited.insert(next);
            let neighbors = self.get_neighbors(next);
            let mut neighbor_distances: Vec<_> = neighbors
                .iter() // Remove empty cells and previously visited nodes
                .filter(|n| n.0 != !0 && !visited.contains(*n))
                .map(|n| {
                    let distance = self
                        .comparator
                        .compare_vec(v.clone(), AbstractVector::Stored(self.get_vector(*n)));
                    (*n, distance)
                })
                .collect();
            neighbor_distances.sort_by_key(|(n, distance)| (OrderedFloat(*distance), *n));

            //eprintln!("calculated neighbor_distances@{next:?}: {neighbor_distances:?}");
            visited.extend(neighbor_distances.iter().map(|(n, _)| n));

            visit_queue.extend(neighbor_distances.iter().enumerate().map(|(ix, (n, d))| {
                (
                    *n,
                    *d,
                    NodeDistance {
                        hops: node_distance.hops + 1,
                        index_sum: node_distance.index_sum + ix + 1,
                    },
                )
            }));
            //eprintln!("before");
            //dbg!(&candidates.data);
            //dbg!(&candidates.priorities);

            let current_best = candidates.first();
            let did_something = candidates.merge_pairs(&neighbor_distances);
            if current_best != candidates.first() {
                // an improvement was made
                highest_improvement = node_distance.index_sum;
            }
            //eprintln!("after");

            if !did_something {
                probe_depth -= 1;
                if probe_depth == 0 {
                    break;
                }
            }
            //dbg!(&candidates.data);
            //dbg!(&candidates.priorities);

            // Sort in reverse order
            visit_queue
                .sort_by_key(|(n, distance, _)| (OrderedFloat(-*distance), (usize::MAX - n.0)))
        }

        highest_improvement
    }

    pub fn closest_vectors<F: Fn(VectorId) -> bool>(
        &self,
        v: AbstractVector<C::T>,
        candidates: &PriorityQueue<VectorId>,
        candidate_count: usize,
        probe_depth: usize,
        include: F,
    ) -> (Vec<(VectorId, f32)>, usize) {
        let pairs: Vec<(NodeId, f32)> = candidates
            .iter()
            // We should only be proceeding downwards!
            .map(|(v, d)| (self.get_node(v).unwrap(), d))
            .collect();
        //eprintln!("pairs: {pairs:?}");
        let mut queue = PriorityQueue::new(candidates.capacity());
        //let mut queue = PriorityQueue::new(candidate_count);
        queue.merge_pairs(&pairs);
        let index_distance = self.closest_nodes(v, &mut queue, probe_depth);
        (
            queue
                .iter()
                .map(|(node_id, distance)| (self.get_vector(node_id), distance))
                .filter(|(v, _)| include(*v))
                .take(candidate_count)
                .collect(),
            index_distance,
        )
    }

    pub fn group_nodes_by_vectors(
        &self,
        vectors: &[VectorId],
    ) -> HashMap<VectorId, Vec<(NodeId, f32)>> {
        // partition nodes by distance to supers
        let mut distances_to_supers: Vec<_> = self
            .nodes
            .par_iter()
            .enumerate()
            .map(|(ix, v)| {
                let node = NodeId(ix);
                let super_distances = (0..vectors.len()).map(|ix2| {
                    let super_vec = vectors[ix2];
                    (
                        super_vec,
                        self.comparator.compare_vec(
                            AbstractVector::Stored(*v),
                            AbstractVector::Stored(super_vec),
                        ),
                    )
                });
                let best_super = super_distances
                    .min_by_key(|(_, d)| OrderedFloat(*d))
                    .unwrap();

                (best_super.0, node, best_super.1)
            })
            .collect();

        distances_to_supers.sort_by_key(|(s, _, _)| s.0);
        let partitions: HashMap<_, _> = distances_to_supers
            .into_iter()
            .group_by(|(s, _, _)| *s)
            .into_iter()
            .map(|(super_vec, group)| {
                let nodes: Vec<_> = group.map(|(_, n, d)| (n, d)).collect();

                (super_vec, nodes)
            })
            .collect();

        partitions
    }

    pub fn multi_node_distances<const N: usize>(
        &self,
        supers: &[VectorId],
    ) -> Vec<[(VectorId, NodeDistance); N]> {
        let mut visit_queue = Vec::with_capacity(supers.len());
        visit_queue.extend(supers.iter().map(|s| (0, self.get_node(*s).unwrap(), *s)));
        let mut result = vec![[(VectorId::MAX, NodeDistance::MAX); N]; self.node_count()];
        for (_, node, super_vec) in visit_queue.iter() {
            let result_entry = &mut result[node.0];
            result_entry[0] = (
                *super_vec,
                NodeDistance {
                    hops: 0,
                    index_sum: 0,
                },
            );
        }

        let mut generation = 1;
        while !visit_queue.is_empty() {
            let mut next_visit_queue = Vec::new();
            // do stuff
            for (_, node, super_vec) in visit_queue {
                let current_distance = result[node.0]
                    .iter()
                    .find(|(v, _)| *v == super_vec)
                    .unwrap()
                    .1;
                for (ix, neighbor) in self
                    .get_neighbors(node)
                    .iter()
                    .filter(|n| !n.is_empty())
                    .enumerate()
                {
                    if !result[neighbor.0].iter().any(|(v, _)| *v == super_vec) {
                        if let Some(insert_pos) = result[neighbor.0]
                            .iter_mut()
                            .find(|(v, _)| *v == VectorId::MAX)
                        {
                            insert_pos.0 = super_vec;
                            insert_pos.1 = NodeDistance {
                                hops: generation,
                                index_sum: current_distance.index_sum + ix + 1,
                            };

                            next_visit_queue.push((insert_pos.1.index_sum, *neighbor, super_vec));
                        }
                    }
                }

                generation += 1;
            }

            next_visit_queue.sort_by_key(|(distance, _, _)| *distance);

            visit_queue = next_visit_queue;
        }
        eprintln!("generation {generation}");

        result
    }

    pub fn node_distances_from_closest_super(&self, supers: &[VectorId]) -> Vec<NodeDistance> {
        let super_groups = self.group_nodes_by_vectors(supers);
        assert_eq!(
            super_groups.values().map(|g| g.len()).sum::<usize>(),
            self.node_count()
        );
        let distances = self.multi_node_distances::<5>(supers);
        let borrowed_distances = &distances;

        let mut distances: Vec<_> = super_groups
            .into_iter()
            .flat_map(|(super_vec, group)| {
                group.into_iter().map(move |(node, _)| {
                    if let Some((_, d)) = borrowed_distances[node.0]
                        .iter()
                        .find(|(v, _)| *v == super_vec)
                    {
                        (node, *d)
                    } else {
                        (node, NodeDistance::MAX)
                    }
                })
            })
            .collect();
        distances.sort_by_key(|(n, _)| *n);

        distances.into_iter().map(|(_, d)| d).collect()
    }

    pub fn nodes_not_connected_to_super(&self, supers: &[VectorId]) -> Vec<NodeId> {
        let distances = self.node_distances_from_closest_super(supers);
        distances
            .into_iter()
            .enumerate()
            .filter(|(_, d)| d.hops == usize::MAX)
            .map(|(n, _)| NodeId(n))
            .collect()
    }

    /// Find the distance of each node to any supernode
    pub fn node_distances(&self, supers: &[VectorId]) -> Vec<NodeDistance> {
        let mut visit_queue = Vec::with_capacity(supers.len());
        visit_queue.extend(supers.iter().map(|s| self.get_node(*s).unwrap()));

        let mut result: Vec<AtomicNodeDistance> = Vec::with_capacity(self.node_count());
        for _ in 0..self.node_count() {
            result.push(AtomicNodeDistance::new());
        }
        for n in visit_queue.iter() {
            let AtomicNodeDistance { index_sum, .. } = &result[n.0];
            index_sum.store(0, atomic::Ordering::Relaxed);
        }

        let mut generation = 0;
        loop {
            visit_queue = visit_queue
                .into_iter()
                .flat_map(|node| {
                    let AtomicNodeDistance {
                        hops,
                        index_sum: distance,
                    } = &result[node.0];
                    let swap_result = hops.compare_exchange(
                        usize::MAX,
                        generation,
                        atomic::Ordering::Relaxed,
                        atomic::Ordering::Relaxed,
                    );
                    let value_after_swap = match swap_result.as_ref() {
                        Ok(_) => generation,
                        Err(v) => *v,
                    };
                    if value_after_swap == generation {
                        let neighbors = self.get_neighbors(node);
                        for (ix, neighbor) in
                            neighbors.iter().enumerate().filter(|(_, n)| n.0 != !0)
                        {
                            let total_distance = distance.load(atomic::Ordering::Relaxed) + ix + 1;
                            let AtomicNodeDistance { index_sum, .. } = &result[neighbor.0];
                            index_sum
                                .fetch_update(
                                    atomic::Ordering::Relaxed,
                                    atomic::Ordering::Relaxed,
                                    |current| Some(usize::min(current, total_distance)),
                                )
                                .unwrap();
                        }

                        if swap_result.is_ok() {
                            itertools::Either::Left(neighbors.iter().cloned().filter(|n| n.0 != !0))
                        } else {
                            itertools::Either::Right(std::iter::empty())
                        }
                    } else {
                        itertools::Either::Right(std::iter::empty())
                    }
                })
                .collect();

            if visit_queue.is_empty() {
                break;
            }
            generation += 1;
        }

        result.into_iter().map(|r| r.finalize()).collect()
    }

    pub fn reachables_from(&self, node: NodeId, check: &[NodeId]) -> Vec<(NodeId, usize)> {
        let mut set: HashSet<NodeId> = HashSet::with_capacity(check.len());
        set.extend(check);
        let mut result = vec![(node, 0)];
        let mut visit_queue = vec![(node, 0)];
        while let Some((node, distance)) = visit_queue.pop() {
            let neighbors = self.get_neighbors(node);
            for (ix, n) in neighbors.iter().enumerate() {
                if n.0 == !0 {
                    continue;
                }
                if set.remove(n) {
                    let new_distance = distance + ix + 1;
                    visit_queue.push((*n, new_distance));
                    result.push((*n, new_distance));
                }
            }
        }

        result
    }

    pub fn discover_nodes_to_promote(&self, supers: &[VectorId]) -> Vec<NodeId> {
        let bottom_distances: Vec<NodeDistance> = self.node_distances(supers);

        let mut bottom_distances: Vec<(NodeId, usize, usize)> = bottom_distances
            .into_iter()
            .enumerate()
            .map(|(ix, d)| (NodeId(ix), d.index_sum, d.hops))
            .collect();
        bottom_distances.sort_by_key(|(n, d, h)| (usize::MAX - d, usize::MAX - h, *n));

        let to_promote: Vec<_> = bottom_distances
            .iter()
            .take_while(|(_, _, d)| *d == usize::MAX)
            .map(|(n, _, _)| *n)
            .collect();

        // TODO! Use buget to increase this with the tail from bottom_distances
        let unreachable = bottom_distances
            .iter()
            .take_while(|(_, _, d)| *d == !0)
            .count();
        let less_than_reachable = to_promote.len() - unreachable;
        let total = to_promote.len();

        eprintln!("promoting {total} vectors ({unreachable} fully unreachable, and {less_than_reachable} less-than-reachables)");
        to_promote
    }

    pub fn reverse_get_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        for n in 0..self.node_count() {
            if self.get_neighbors(NodeId(n)).contains(&node) {
                result.push(NodeId(n));
            }
        }

        result
    }
}

struct AtomicNodeDistance {
    hops: AtomicUsize,
    index_sum: AtomicUsize,
}

impl AtomicNodeDistance {
    fn new() -> Self {
        Self {
            hops: AtomicUsize::new(usize::MAX),
            index_sum: AtomicUsize::new(usize::MAX),
        }
    }

    fn finalize(self) -> NodeDistance {
        unsafe { std::mem::transmute(self) }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash, Default)]
pub struct NodeDistance {
    pub hops: usize,
    pub index_sum: usize,
}

impl NodeDistance {
    const ZERO: Self = Self {
        hops: 0,
        index_sum: 0,
    };
    const MAX: Self = Self {
        hops: usize::MAX,
        index_sum: usize::MAX,
    };
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Hnsw<C> {
    pub layers: Vec<Layer<C>>,
    order: usize,
    neighborhood_size: usize,
    zero_layer_neighborhood_size: usize,
}

impl<C> Hnsw<C> {
    pub fn vector_count(&self) -> usize {
        self.get_layer(0).map(|l| l.node_count()).unwrap_or(0)
    }

    pub fn neighborhood_size(&self) -> usize {
        self.neighborhood_size
    }

    pub fn zero_neighborhood_size(&self) -> usize {
        self.zero_layer_neighborhood_size
    }

    pub fn get_layer(&self, i: usize) -> Option<&Layer<C>> {
        self.get_layer_from_top(self.layers.len() - i - 1)
    }

    pub fn get_layer_from_top_mut(&mut self, i: usize) -> Option<&mut Layer<C>> {
        if i < self.layer_count() {
            Some(&mut self.layers[i])
        } else {
            // eprintln!("No layer");
            None
        }
    }

    pub fn get_layer_from_top(&self, i: usize) -> Option<&Layer<C>> {
        if i < self.layer_count() {
            Some(&self.layers[i])
        } else {
            // eprintln!("No layer");
            None
        }
    }

    #[allow(unused)]
    fn layer_from_top_to_layer(&self, layer: usize) -> usize {
        self.layer_count() - layer - 1
    }

    pub fn get_layer_above(&self, i: usize) -> Option<&Layer<C>> {
        if i == 0 {
            None
        } else {
            self.get_layer_from_top(i - 1)
        }
    }

    pub fn entry_vector(&self) -> VectorId {
        // Other choices are possible
        VectorId(0)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn comparator(&self) -> &C {
        &self.layers[0].comparator
    }
}

impl<C: Comparator + 'static> Hnsw<C> {
    pub fn search(
        &self,
        v: AbstractVector<C::T>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        search::search_layers(v, number_of_candidates, &self.layers, probe_depth, None)
    }

    pub fn search_noisy(
        &self,
        v: AbstractVector<C::T>,
        number_of_candidates: usize,
        probe_depth: usize,
        noisy: bool,
    ) -> (Vec<(VectorId, f32)>, usize) {
        search::search_layers_noisy(
            v,
            number_of_candidates,
            &self.layers,
            probe_depth,
            noisy,
            None,
        )
    }

    pub fn generate_layer(
        &self,
        comparator: C,
        vs: Vec<VectorId>,
        neighborhood_size: usize,
        new_top: bool,
    ) -> Layer<C> {
        assert!(!vs.is_empty(), "tried to construct an empty layer");
        // Parameter for the number of neighbours to look at from the proceeding layer.
        let number_of_supers_to_check = 6; //neighborhood_size;

        eprintln!("Constructing neighorhood data structure");
        // This is the final neighbors array, we are going to slice it up here
        // and seed it with some prospects.
        let total_size = vs.len() * neighborhood_size;
        let mut neighbors = vec![NodeId(!0); total_size];
        // This is an auxilliary to keep track of the neighbors quality in a neighborhood
        let mut neighbor_distances = vec![f32::MAX; total_size];

        // This was needed to reduce memory pressure
        // we should *really* consider succinct data structures
        eprintln!("Finding partition groups");
        // 1. Calculate our node id, and find our neighborhood in the above layer
        let layers: &[Layer<_>] = if new_top { &[] } else { &self.layers };
        let initial_partitions = search::generate_initial_partitions(
            &vs,
            &comparator,
            number_of_supers_to_check,
            layers,
            0,
        );

        eprintln!("Generating partition groups");
        // 2. Partition the layer in terms of the closeness to the
        // best node in the layer above
        let partition_groups = initial_partitions
            .into_iter()
            .into_group_map_by(|(_, _, distances)| distances.first().map(|(id, _)| *id));

        // 3. Calculate our neighbourhoods by comparing distances in our partition
        let borrowed_comparator = &comparator;

        eprintln!("Scanning partition groups");
        partition_groups.par_iter().for_each(|(_sup, partition)| {
            partition
                .par_iter()
                .for_each(|(node_id, vector_id, distances)| {
                    let mut distances = distances.clone();
                    let super_nodes: Vec<_> =
                        distances.iter().map(|(node, _)| node).cloned().collect();

                    // some random, some for neighborhood
                    // TODO - also some random extra nodes on the same layer
                    let mut prng = StdRng::seed_from_u64(
                        self.layer_count() as u64 + vector_id.0 as u64 + vs.len() as u64,
                    );

                    let mut partitions: Vec<_> = super_nodes
                        .into_iter()
                        .filter_map(|n| partition_groups.get(&Some(n)))
                        .collect();
                    if partitions.is_empty() {
                        // probably we're in the top layer. best add ourselves.
                        partitions.push(partition);
                    }
                    let partition_maxes: Vec<_> = partitions.iter().map(|p| p.len()).collect();

                    let choice_count =
                        std::cmp::min(neighborhood_size * 5, partition_maxes.iter().sum());
                    let partition_choices =
                        choose_n(choice_count, partition_maxes, node_id.0, &mut prng);

                    for i in 0..partition_choices.len() {
                        let partition = partitions[partition_choices[i].0];
                        let choice = &partition[partition_choices[i].1];
                        let distance = borrowed_comparator.compare_vec(
                            AbstractVector::Stored(*vector_id),
                            AbstractVector::Stored(choice.1),
                        );
                        distances.push((choice.0, distance));
                    }
                    distances.sort_by_key(|d| (OrderedFloat(d.1), d.0));
                    distances.dedup();
                    let (mut nodes, mut distances): (Vec<_>, Vec<f32>) = distances
                        .into_iter()
                        .filter(|(n, _d)| node_id != n)
                        .take(neighborhood_size)
                        .unzip();
                    //eprintln!("nodes@{node_id:?}: {nodes:?}");
                    nodes.resize_with(neighborhood_size, || NodeId(!0));
                    distances.resize_with(neighborhood_size, || f32::MAX);

                    let unsafe_nodes = unsafe {
                        std::slice::from_raw_parts_mut(
                            neighbors.as_ptr().add(node_id.0 * neighborhood_size) as *mut NodeId,
                            neighborhood_size,
                        )
                    };
                    let unsafe_distances = unsafe {
                        std::slice::from_raw_parts_mut(
                            neighbor_distances
                                .as_ptr()
                                .add(node_id.0 * neighborhood_size)
                                as *mut f32,
                            neighborhood_size,
                        )
                    };

                    unsafe_nodes.copy_from_slice(&nodes);
                    unsafe_distances.copy_from_slice(&distances);
                });
        });

        let neighbor_candidates: Vec<RwLock<PriorityQueue<NodeId>>> = neighbors
            .chunks_exact_mut(neighborhood_size)
            .zip(neighbor_distances.chunks_exact_mut(neighborhood_size))
            .map(|(data, priorities)| RwLock::new(PriorityQueue::from_slices(data, priorities)))
            .collect();

        eprintln!("Making neighborhoods bidirectional");
        // 4. Make neighborhoods bidirectional
        (0..neighbor_candidates.len())
            .into_par_iter()
            .for_each(|i| {
                let node = NodeId(i);
                let neighborhood_copy: Vec<(NodeId, f32)> = neighbor_candidates[i]
                    .read()
                    .unwrap()
                    .iter()
                    .filter(|(n, _)| n.0 != !0)
                    .collect();
                //eprintln!("{neighborhood_copy:?}");
                for (neighbor, distance) in neighborhood_copy {
                    //eprintln!("inserting into: {}", neighbor.0);
                    neighbor_candidates[neighbor.0]
                        .write()
                        .unwrap()
                        .insert(node, distance);
                }
            });

        Layer {
            neighborhood_size,
            comparator,
            nodes: vs,
            neighbors,
        }
    }

    pub fn generate(
        c: C,
        vs: Vec<VectorId>,
        neighborhood_size: usize,
        zero_layer_neighborhood_size: usize,
        order: usize,
    ) -> Self {
        let total_size = vs.len();
        assert!(total_size > 0);
        // eprintln!("neighborhood_size: {neighborhood_size}");
        // eprintln!("total_size: {total_size}");
        // eprintln!("layer count: {layer_count}");
        let partitions = calculate_partitions(total_size, order);
        assert!(!partitions.is_empty());
        let layer_count = partitions.len();
        let layers = Vec::with_capacity(layer_count);
        let mut hnsw: Hnsw<C> = Hnsw {
            layers,
            neighborhood_size,
            zero_layer_neighborhood_size,
            order,
        };
        for (i, length) in partitions.iter().enumerate() {
            let level = layer_count - i - 1;
            let slice_length = std::cmp::min(*length, total_size);
            let slice = &vs[0..slice_length];
            let neighbors = if level == 0 {
                zero_layer_neighborhood_size
            } else {
                neighborhood_size
            };
            let layer = hnsw.generate_layer(c.clone(), slice.to_vec(), neighbors, false);
            hnsw.layers.push(layer);
            eprintln!("linking to better neighbors (during construction)");
            hnsw.improve_neighbors(0.01, 0.01, None);
        }

        hnsw
    }

    pub fn len(&self) -> usize {
        self.get_layer(0).unwrap().node_count()
    }

    pub fn is_empty(&self) -> bool {
        self.get_layer(0)
            .map(|l| l.nodes.is_empty())
            .unwrap_or(false)
    }

    pub fn knn(
        &self,
        k: usize,
        probe_depth: usize,
    ) -> impl ParallelIterator<Item = (VectorId, Vec<(VectorId, f32)>)> + '_ {
        let layer = &self.layers[self.layers.len() - 1];
        let nodes = &layer.nodes;

        nodes.par_iter().enumerate().map(move |(i, v)| {
            let node = NodeId(i);
            let abstract_vector = AbstractVector::Stored(*v);
            let eff_factor = 3;
            let mut pq = PriorityQueue::new(k * eff_factor);
            pq.merge_pairs(&[(node, 0.0)]);
            layer.closest_nodes(abstract_vector, &mut pq, probe_depth);
            let distances: Vec<_> = pq
                .iter()
                .filter(|(n, _)| *n != node)
                .map(|(node_id, distance)| (layer.get_vector(node_id), distance))
                .take(k)
                .collect();
            (*v, distances)
        })
    }

    pub fn threshold_nn(
        &self,
        threshold: f32,
        probe_depth: usize,
        initial_search_depth: usize,
    ) -> impl IndexedParallelIterator<Item = (VectorId, Vec<(VectorId, f32)>)> + '_ {
        let layer = &self.layers[self.layers.len() - 1];
        let nodes = &layer.nodes;

        nodes.par_iter().enumerate().map(move |(i, v)| {
            let node = NodeId(i);
            let abstract_vector = AbstractVector::Stored(*v);
            let mut pq = PriorityQueue::new(initial_search_depth);
            pq.merge_pairs(&[(node, 0.0)]);
            let mut last = 0.0;
            let mut last_size = 0;
            while last < threshold && pq.len() > last_size {
                last_size = pq.len();
                layer.closest_nodes(abstract_vector.clone(), &mut pq, probe_depth);
                last = pq.last().expect("should have at least retrieved self").1;
                if last < threshold && pq.len() == pq.capacity() {
                    pq.resize_capacity(pq.capacity() * 2);
                }
            }
            let distances: Vec<_> = pq
                .iter()
                .filter(|(n, _)| *n != node)
                .take_while(|(_n, distance)| *distance < threshold)
                .map(|(node_id, distance)| (layer.get_vector(node_id), distance))
                .collect();
            (*v, distances)
        })
    }

    pub fn par_all_vectors(&self) -> impl ParallelIterator<Item = VectorId> + '_ {
        self.get_layer(0).unwrap().nodes.par_iter().cloned()
    }

    pub fn all_vectors(&self) -> AllVectorIterator {
        self.get_layer(0)
            .map(|layer| {
                let iter = layer.nodes.iter();
                AllVectorIterator::Full { iter }
            })
            .unwrap_or(AllVectorIterator::Empty)
    }

    pub fn supers_for_layer(&self, layer_id: usize) -> &[VectorId] {
        if self.layer_count() == layer_id + 1 {
            let layer = self.get_layer(layer_id).unwrap();
            &layer.nodes[0..1]
        } else {
            &self.get_layer(layer_id + 1).unwrap().nodes[..]
        }
    }

    pub fn node_distances_for_layer(&self, layer_id: usize) -> Vec<NodeDistance> {
        let layer = self.get_layer(layer_id).unwrap();
        let supers = self.supers_for_layer(layer_id);
        layer.node_distances(supers)
    }

    pub fn reachables_from_node_for_layer(
        &self,
        layer_id: usize,
        node: NodeId,
        check: &[NodeId],
    ) -> Vec<(NodeId, usize)> {
        let layer = self.get_layer_from_top(layer_id).unwrap();
        layer.reachables_from(node, check)
    }

    pub fn discover_vectors_to_promote(&self, layer_id_from_top: usize) -> Vec<VectorId> {
        //const THRESHOLD: usize = 42;
        let layers = &self.layers[0..=layer_id_from_top];
        let layer_above = if layer_id_from_top == 0 {
            None
        } else {
            Some(&layers[layer_id_from_top - 1])
        };
        let current_layer = &layers[layer_id_from_top];
        current_layer
            .nodes
            .par_iter()
            .filter_map(|vector| {
                let (matches, _index_distance) = search::search_layers_noisy(
                    AbstractVector::Stored(*vector),
                    300,
                    layers,
                    2,
                    false,
                    None,
                );
                let vector_is_in_matches = matches[0].0 == *vector;
                if !vector_is_in_matches
                    && (layer_above.is_none() || layer_above.unwrap().get_node(*vector).is_none())
                {
                    Some(*vector)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn extend_layer(&mut self, layer_id: usize, vecs: Vec<VectorId>) {
        let layer_id_from_top = self.layer_count() - layer_id - 1;
        eprintln!("Extending layer: {layer_id:?}");
        eprintln!("Counting as {layer_id_from_top:?}");
        let (layers_above, layers_below) = self.layers.split_at_mut(layer_id_from_top);
        eprintln!(
            "layers above: {}, layers_below: {}",
            layers_above.len(),
            layers_below.len()
        );
        let layer = &mut layers_below[0];

        let (old_nodes, old_nodes_map, new_nodes_map, _vecs) = generate_node_maps(vecs, layer);

        eprintln!("created node maps");
        assert_eq!(old_nodes_map.len() + new_nodes_map.len(), layer.nodes.len());

        let new_neighbors_len = layer.nodes.len() * layer.neighborhood_size;
        let mut old_neighbors = Vec::with_capacity(new_neighbors_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            old_neighbors.set_len(new_neighbors_len)
        };
        eprintln!("allocated new neighbors");
        std::mem::swap(&mut layer.neighbors, &mut old_neighbors);

        copy_old_neighborhoods_into_layer(&old_nodes, &old_neighbors, &old_nodes_map, layer);
        eprintln!("copied old neighbors into new layer");
        initialize_new_neighborhoods_into_layer(&new_nodes_map, layer);
    }

    pub fn link_layer_to_better_neighbors(&mut self, layer_from_top: usize) -> usize {
        self.link_nodes_in_layer_to_better_neighbors(
            layer_from_top,
            (0..self.get_layer_from_top(layer_from_top).unwrap().nodes.len())
                .into_par_iter()
                .map(NodeId),
        )
    }

    pub fn link_nodes_in_layer_to_better_neighbors<
        I: ParallelIterator<Item = NodeId>,
        II: IntoParallelIterator<Iter = I, Item = NodeId>,
    >(
        &mut self,
        layer_from_top: usize,
        nodes: II,
    ) -> usize {
        let (top_stack, bottom_stack) = self.layers.split_at_mut(layer_from_top);
        let current_layer = &mut bottom_stack[0];

        let pseudo_layer = current_layer.clone();
        let mut pseudo_stack: Vec<&Layer<C>> = Vec::with_capacity(top_stack.len() + 1);
        pseudo_stack.extend(top_stack.iter());
        pseudo_stack.push(&pseudo_layer);

        let neighbor_candidates: Vec<RwLock<&mut [NodeId]>> = current_layer
            .neighbors
            .chunks_exact_mut(current_layer.neighborhood_size)
            .map(RwLock::new)
            .collect();
        nodes
            .into_par_iter()
            .map(|local_node| {
                let mut count = 0;
                let vector = pseudo_layer.get_vector(local_node);
                let matches = search::search_layers(
                    AbstractVector::Stored(vector),
                    self.neighborhood_size,
                    &pseudo_stack,
                    1,
                    Some(vector),
                );
                for (neighbor_vec, distance) in matches.into_iter().take(10) {
                    if neighbor_vec == vector {
                        break;
                    }
                    let neighbor = pseudo_layer.get_node(neighbor_vec).unwrap();
                    let neighbor_of_neighbors = neighbor_candidates[neighbor.0].read().unwrap();
                    if let Some(insert_pos) = neighbor_of_neighbors.iter().position(|n| {
                        if n.is_empty() || *n == local_node {
                            return true;
                        }
                        let v = pseudo_layer.get_vector(*n);
                        let other_distance = current_layer.comparator.compare_vec(
                            AbstractVector::Stored(v),
                            AbstractVector::Stored(neighbor_vec),
                        );
                        distance < other_distance || (distance == other_distance && local_node < *n)
                    }) {
                        if neighbor_of_neighbors[insert_pos] == local_node {
                            // we are already in this neighbor list. do not add again
                            continue;
                        }
                        std::mem::drop(neighbor_of_neighbors);
                        let mut neighbor_of_neighbors =
                            neighbor_candidates[neighbor.0].write().unwrap();

                        for i in (insert_pos..neighbor_of_neighbors.len() - 1).rev() {
                            neighbor_of_neighbors[i + 1] = neighbor_of_neighbors[i];
                        }
                        neighbor_of_neighbors[insert_pos] = local_node;
                        count += 1;
                    }
                }

                count
            })
            .sum()
    }

    fn improve_neighborhoods_at_layer(&mut self, layer_from_top: usize) -> usize {
        eprintln!("improving {layer_from_top}");
        let count = self.link_layer_to_better_neighbors(layer_from_top);
        eprintln!("{layer_from_top}: relinked {count}");
        count
    }

    pub fn promote_at_layer(&mut self, layer_from_top: usize, max_proportion: f32) -> bool {
        eprintln!("promoting batch at layer from top: {layer_from_top}");
        let mut vecs = self.discover_vectors_to_promote(layer_from_top);
        if vecs.is_empty() {
            return false;
        }
        if max_proportion < 1.0 {
            let vec_length = vecs.len();
            vecs = vecs.into_iter().take(vec_length).collect();
        }
        eprintln!(
            "vec len for promotion: {}@{} (out of {})",
            vecs.len(),
            layer_from_top,
            self.get_layer_from_top(layer_from_top)
                .unwrap()
                .node_count()
        );
        vecs.sort();
        if layer_from_top == 0 {
            // just construct a new hnsw and copy over layer stack
            let new_top = Self::generate(
                self.comparator().clone(),
                vecs.clone(),
                self.neighborhood_size,
                self.neighborhood_size,
                self.order,
            );
            let mut layers = new_top.layers;
            eprintln!(
                "generated {} new top layers (and nothing else)",
                layers.len()
            );
            std::mem::swap(&mut self.layers, &mut layers);
            self.layers.extend(layers);
            for layer in self.layers.iter() {
                eprintln!("layer count: {}", layer.node_count());
            }
        } else {
            let mut sizes: Vec<_> = self
                .layers
                .iter()
                .take(layer_from_top)
                .map(|l| l.node_count())
                .collect();
            sizes.reverse();
            eprintln!("sizes: {sizes:?}");
            let mut promotions = calculate_partitions_for_additions(&sizes, vecs.len(), self.order);
            eprintln!("promotions: {promotions:?}");
            let mut new_top_len = 0;
            if promotions.len() > layer_from_top {
                // we are going to need at least one more layer
                let count = promotions[layer_from_top];
                let top_vecs: Vec<_> = vecs.iter().cloned().take(count).collect();
                let new_top = Self::generate(
                    self.comparator().clone(),
                    top_vecs.clone(),
                    self.neighborhood_size,
                    self.neighborhood_size,
                    self.order,
                );
                let mut layers = new_top.layers;
                new_top_len = layers.len();
                eprintln!("generated {} new top layers (and extending)", layers.len());
                std::mem::swap(&mut self.layers, &mut layers);
                self.layers.extend(layers);
                for layer in self.layers.iter() {
                    eprintln!("layer count: {}", layer.node_count());
                }
            }

            promotions.truncate(layer_from_top);
            eprintln!("truncated promotions: {promotions:?}");
            promotions.reverse();

            eprintln!("promotion sizes after maybe having generated a top {promotions:?}");

            for (original_layer_id_from_top, promotion_count) in promotions.into_iter().enumerate()
            {
                if promotion_count == 0 {
                    continue;
                }
                eprintln!("original_layer_id_from_top: {original_layer_id_from_top}");
                let layer_id_from_top = original_layer_id_from_top + new_top_len;
                eprintln!("layer_id_from_top: {layer_id_from_top}");
                let layer = self.get_layer_from_top(layer_id_from_top).unwrap();
                let vecs_to_promote: Vec<_> = vecs
                    .iter()
                    .cloned()
                    .take(promotion_count)
                    .filter(|v| layer.get_node(*v).is_none())
                    .collect();

                let layer_id_from_bottom = self.layer_from_top_to_layer(layer_id_from_top);
                self.extend_layer(layer_id_from_bottom, vecs_to_promote)
            }
        }
        true
    }

    /*
    #[allow(unused)]
    fn promote_at_layer(&mut self, layer_from_top: usize) -> (Vec<VectorId>, usize) {
        let mut vecs = self.discover_vectors_to_promote_2(layer_from_top);
        vecs.sort();
        let count = vecs.len();
        eprintln!("layer_from_top {layer_from_top}: promoting {count} vecs");
        //eprintln!("Vectors to promote: {vecs:?}");
        if count != 0 {
            if layer_from_top == 0 {
                // generate new layer(s)
                let new_top = Self::generate(
                    self.comparator().clone(),
                    vecs.clone(),
                    self.neighborhood_size,
                    self.neighborhood_size,
                    self.order,
                );
                let mut layers = new_top.layers;
                eprintln!("generated {} new top layers", layers.len());
                std::mem::swap(&mut self.layers, &mut layers);
                self.layers.extend(layers);
                return (vecs, 0);
            } else {
                // extend existing layer
                let layer_above = self.layer_from_top_to_layer(layer_from_top - 1);
                self.extend_layer(layer_above, vecs.clone());
                return (vecs, layer_from_top - 1);
            }
        }

        (vecs, layer_from_top)
    }
    */
    pub fn stochastic_recall_upto(&self, upto: usize, recall_proportion: f32) -> f32 {
        let mut rng = StdRng::seed_from_u64(42);
        eprintln!("stochastic recall upto: {upto}");
        let layer = self.get_layer_from_top(upto).unwrap();
        let total = layer.node_count();
        let selection = usize::max(1, (total as f32 * recall_proportion) as usize);
        let vecs_to_find: Vec<VectorId> = if selection == total {
            layer.nodes.clone()
        } else {
            let mut vecs = layer.nodes.clone();
            vecs.shuffle(&mut rng);
            vecs.truncate(selection);
            vecs
        };
        let relevant: usize = vecs_to_find
            .par_iter()
            .filter(|vid| {
                let res = self.search(AbstractVector::Stored(**vid), 300, 2);
                res.iter().map(|(v, _)| v).any(|v| v == *vid)
            })
            .count();

        relevant as f32 / selection as f32
    }

    pub fn stochastic_recall(&self, recall_proportion: f32) -> f32 {
        self.stochastic_recall_upto(self.layer_count() - 1, recall_proportion)
    }

    pub fn recall(&self) -> f32 {
        let total = self.len();
        let total_relevant: usize = self
            .par_all_vectors()
            .map(|vector_id| {
                let vector = self.comparator().lookup(vector_id);
                let vector_data = &*vector;
                /* eprintln!("XXXXXXXXXXXXXXXXXXXXXX");
                eprintln!("Searching for {i}");
                */
                let v = AbstractVector::Unstored(vector_data);
                let results = self.search(v, 300, 2);
                if vector_id == results[0].0 {
                    1
                } else {
                    0
                }
            })
            .sum();

        total_relevant as f32 / total as f32
    }

    pub fn improve_neighbors(
        &mut self,
        threshold: f32,
        recall_proportion: f32,
        last_recall: Option<f32>,
    ) -> f32 {
        self.improve_neighbors_upto(
            self.layer_count(),
            threshold,
            recall_proportion,
            last_recall,
        )
    }

    pub fn improve_neighbors_upto(
        &mut self,
        upto: usize,
        threshold: f32,
        recall_proportion: f32,
        last_recall: Option<f32>,
    ) -> f32 {
        assert!(upto <= self.layer_count());
        let mut last_recall = last_recall.unwrap_or(0.0);
        let mut last_improvement = 1.0_f32;
        while last_improvement >= threshold && last_recall < 1.0 {
            for layer_id_from_top in 0..upto {
                let count = self.improve_neighborhoods_at_layer(layer_id_from_top);
                eprintln!("layer {layer_id_from_top}: improved {count}");
            }
            let recall = self.stochastic_recall_upto(upto - 1, recall_proportion);
            last_improvement = recall - last_recall;
            last_recall = recall;
            eprintln!("recall {recall} (improvement: {last_improvement})");
        }

        last_recall
    }

    pub fn improve_index(
        &mut self,
        promotion_threshold: f32,
        neighbor_threshold: f32,
        recall_proportion: f32,
        last_recall: Option<f32>,
    ) -> f32 {
        // let's start with a neighborhood optimization so we don't overpromote
        let mut recall = last_recall.unwrap_or_else(|| self.stochastic_recall(recall_proportion));
        //let mut last_recall =
        //    self.improve_neighbors(neighbor_threshold, recall_proportion, Some(recall));

        let mut improvement = 1.0;
        let mut bailout = 20;
        while improvement >= promotion_threshold && recall < 1.0 && bailout != 0 {
            let last_recall = recall;
            let mut upto = 0;
            while upto < self.layer_count() && bailout != 0 {
                let layer_count = self.layer_count();
                eprintln!("About to promote");
                if self.promote_at_layer(upto, promotion_threshold) {
                    // promotion might have changed the layer count by adding new top layers.
                    // do a separate recall measure, cause recall might
                    // have dropped with this promotion and we don't want
                    // it to 'count' as a termination condition in the improve_neighbors.
                    recall = self.stochastic_recall_upto(upto, recall_proportion);

                    recall = self.improve_neighbors_upto(
                        upto,
                        neighbor_threshold,
                        recall_proportion,
                        Some(recall),
                    );
                    eprintln!("recall after promotion: {recall}");
                }
                if self.layer_count() > layer_count {
                    bailout -= 1;
                    upto = 0;
                } else {
                    upto += 1;
                }
            }
            bailout -= 1;
            improvement = recall - last_recall;
            eprintln!("outer loop improvement: {improvement}");
        }

        recall
    }
}

impl<C: Serializable + Clone> Hnsw<C> {
    pub fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        serialize::serialize_hnsw(
            self.neighborhood_size,
            self.zero_layer_neighborhood_size,
            self.order,
            &self.layers,
            path,
        )
    }

    pub fn deserialize<P: AsRef<Path>>(
        path: P,
        params: C::Params,
    ) -> Result<Self, SerializationError> {
        serialize::deserialize_hnsw(path, params)
    }
}

fn cross_compare_vectors<C: Comparator + 'static>(
    vecs: &Vec<VectorId>,
    borrowed_comparator: &C,
) -> HashMap<VectorId, Vec<(VectorId, f32)>> {
    let cross_compare: HashMap<VectorId, Vec<(VectorId, f32)>> = vecs
        .par_iter()
        .map(|v| {
            let distances: Vec<(VectorId, f32)> = vecs
                .par_iter()
                .flat_map(|w| {
                    if w == v {
                        None
                    } else {
                        let distance = borrowed_comparator
                            .compare_vec(AbstractVector::Stored(*w), AbstractVector::Stored(*v));
                        Some((*w, distance))
                    }
                })
                .collect();
            (*v, distances)
        })
        .collect();
    cross_compare
}

fn initialize_new_neighborhoods_into_layer<C: Comparator + 'static>(
    new_nodes_map: &[usize],
    layer: &mut Layer<C>,
) {
    for node in new_nodes_map {
        let neighbors = layer.get_neighbors_mut(NodeId(*node));
        neighbors.fill(NodeId(!0));
    }
}

fn copy_old_neighborhoods_into_layer<C: Comparator + 'static>(
    old_nodes: &[VectorId],
    old_neighbors: &[NodeId],
    old_nodes_map: &[usize],
    layer: &mut Layer<C>,
) {
    // insert old nodes with shifted offsets
    (0..old_nodes.len())
        .into_par_iter()
        .for_each(|old_node_id| {
            let nhs = layer.neighborhood_size;
            let old_neighborhood = &old_neighbors[old_node_id * nhs..(old_node_id + 1) * nhs];
            let new_node_id = old_nodes_map[old_node_id];
            let new_neighborhood = unsafe {
                std::slice::from_raw_parts_mut(
                    layer.neighbors.as_ptr().add(new_node_id * nhs) as *mut NodeId,
                    nhs,
                )
            };

            for i in 0..nhs {
                if old_neighborhood[i].0 == !0 {
                    new_neighborhood[i] = NodeId(!0);
                } else {
                    new_neighborhood[i] = NodeId(old_nodes_map[old_neighborhood[i].0]);
                }
            }
        });
}

fn generate_node_maps<C: Comparator + 'static>(
    mut vecs: Vec<VectorId>,
    layer: &mut Layer<C>,
) -> (Vec<VectorId>, Vec<usize>, Vec<usize>, Vec<VectorId>) {
    vecs.sort();
    // this will be swapped out for the current nodes, therefore the variable name
    let mut old_nodes = Vec::with_capacity(vecs.len() + layer.nodes.len());
    std::mem::swap(&mut old_nodes, &mut layer.nodes);

    let mut old_nodes_iter = old_nodes.iter().peekable();
    let mut new_nodes_iter = vecs.iter().peekable();
    let mut old_nodes_map = Vec::with_capacity(old_nodes.len());
    let mut new_nodes_map = Vec::with_capacity(vecs.len());
    let mut vecs: Vec<VectorId> = Vec::with_capacity(vecs.len());
    loop {
        let old_nodes_next = old_nodes_iter.peek();
        let new_nodes_next = new_nodes_iter.peek();
        match (old_nodes_next, new_nodes_next) {
            (None, None) => break,
            (Some(_), None) => {
                layer.nodes.push(*old_nodes_iter.next().unwrap());
                old_nodes_map.push(layer.nodes.len() - 1);
            }
            (None, Some(v)) => {
                vecs.push(**v);
                layer.nodes.push(*new_nodes_iter.next().unwrap());
                new_nodes_map.push(layer.nodes.len() - 1);
            }
            (Some(old), Some(new)) => match old.cmp(new) {
                std::cmp::Ordering::Equal => {
                    panic!("tried to insert vector that already exists in this layer")
                }
                std::cmp::Ordering::Less => {
                    layer.nodes.push(*old_nodes_iter.next().unwrap());
                    old_nodes_map.push(layer.nodes.len() - 1);
                }
                std::cmp::Ordering::Greater => {
                    vecs.push(**new);
                    layer.nodes.push(*new_nodes_iter.next().unwrap());
                    new_nodes_map.push(layer.nodes.len() - 1);
                }
            },
        }
    }
    (old_nodes, old_nodes_map, new_nodes_map, vecs)
}

pub enum AllVectorIterator<'a> {
    Full { iter: Iter<'a, VectorId> },
    Empty,
}

impl<'a> Iterator for AllVectorIterator<'a> {
    type Item = VectorId;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AllVectorIterator::Full { iter } => iter.next().cloned(),
            AllVectorIterator::Empty => None,
        }
    }
}

fn choose_n_1(
    n: usize,
    partition_maxes: Vec<usize>,
    exclude: usize,
    prng: &mut StdRng,
) -> Vec<(usize, usize)> {
    let mut result: Vec<(usize, usize)> = (0..partition_maxes.len())
        .flat_map(|partition| {
            (0..partition_maxes[partition]).filter_map(move |in_partition| {
                if partition == 0 && in_partition == exclude {
                    None
                } else {
                    Some((partition, in_partition))
                }
            })
        })
        .collect();
    result.shuffle(prng);

    result.truncate(n);

    result
}

fn choose_n(
    n: usize,
    partition_maxes: Vec<usize>,
    exclude: usize,
    prng: &mut StdRng,
) -> Vec<(usize, usize)> {
    // todo: probably should give higher chance to select our own partition
    if partition_maxes.iter().sum::<usize>() * 2 > n {
        return choose_n_1(n, partition_maxes, exclude, prng);
    }

    let mut count = 0;
    let mut set = HashSet::with_capacity(n);
    let exp = Exp::new(1.0_f32).unwrap();
    while count != n {
        let mut which_partition = exp.sample(prng).floor() as usize;
        if which_partition >= partition_maxes.len() {
            which_partition = 0;
        }
        let selection = prng.gen_range(0..partition_maxes[which_partition]);
        if (which_partition != 0 || selection != exclude)
            && set.insert((which_partition, selection))
        {
            count += 1;
        }
    }
    set.into_iter().collect()
}

fn calculate_partitions_from_bottom(total_size: usize, order: usize) -> Vec<usize> {
    let mut partitions: Vec<usize> = vec![];
    let mut size = total_size;
    let layer_count = usize::max(1, (total_size as f32).log(order as f32).ceil() as usize);
    for _ in 0..layer_count {
        partitions.push(size);
        size /= order;
    }

    partitions
}

fn calculate_partitions(total_size: usize, order: usize) -> Vec<usize> {
    let mut partitions = calculate_partitions_from_bottom(total_size, order);
    partitions.reverse();
    partitions
}

fn calculate_partitions_for_additions(
    sizes_from_bottom: &[usize],
    new_vecs: usize,
    order: usize,
) -> Vec<usize> {
    let mut new_partitions =
        calculate_partitions_from_bottom(sizes_from_bottom[0] + new_vecs, order);
    // ensure we have at least as many partitions as existing sizes (but maybe more)
    if new_partitions.len() < sizes_from_bottom.len() {
        new_partitions.resize(sizes_from_bottom.len(), 0);
    }
    eprintln!("new partitions: {new_partitions:?}\nsizes: {sizes_from_bottom:?}");

    let mut new_sizes: Vec<usize> = new_partitions
        .into_iter()
        .enumerate()
        .map(|(ix, p)| {
            if ix >= sizes_from_bottom.len() {
                p
            } else {
                usize::max(p, sizes_from_bottom[ix])
            }
        })
        .collect();

    // Ensure that the new layer stack is monotonic.
    // We have to make sure that we don't cause a super layer to grow larger than its sublayers.
    let mut last = 0_usize;
    for elt in new_sizes.iter_mut().rev() {
        if last > *elt {
            *elt = last;
        }
        last = *elt;
    }
    eprintln!("corrected partitions: {new_sizes:?}");

    let mut partition_additions: Vec<usize> = new_sizes
        .into_iter()
        .enumerate()
        .map(|(ix, p)| {
            if ix >= sizes_from_bottom.len() {
                p
            } else {
                p - sizes_from_bottom[ix]
            }
        })
        .collect();

    // ensure that promotions are reverse monotonic.
    // We have to make sure that if we promote at a superlayer, it is also promoted at the sublayer.
    let mut last = 0_usize;
    for elt in partition_additions.iter_mut().rev() {
        if last > *elt {
            *elt = last;
        }
        last = *elt;
    }
    partition_additions
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use rand_distr::Uniform;

    use super::bigvec::*;
    use super::*;
    type SillyVec = [f32; 3];
    #[derive(Clone, Debug, PartialEq)]
    struct SillyComparator {
        data: Vec<SillyVec>,
    }

    impl Comparator for SillyComparator {
        type T = SillyVec;
        type Borrowable<'a> = &'a SillyVec;

        fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
            &self.data[v.0]
        }

        fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
            let mut result = 0.0;
            for (&f1, &f2) in v1.iter().zip(v2.iter()) {
                result += f1 * f2
            }
            1.0 - result
        }
    }

    fn make_simple_hnsw() -> Hnsw<SillyComparator> {
        let sqrt2_recip = std::f32::consts::FRAC_1_SQRT_2;
        let data: Vec<SillyVec> = vec![
            [1.0, 0.0, 0.0],                 // 0
            [0.0, 1.0, 0.0],                 // 1
            [0.0, 0.0, 1.0],                 // 2
            [sqrt2_recip, sqrt2_recip, 0.0], // 3
            [0.5773, 0.5773, 0.5773],        // 4
            [-1.0, 0.0, 0.0],                // 5
            [0.0, -1.0, 0.0],                // 6
            [0.0, 0.0, -1.0],                // 7
            [0.0, sqrt2_recip, sqrt2_recip], // 8
        ];
        let c = SillyComparator { data: data.clone() };
        let vs: Vec<_> = (0..9).map(VectorId).collect();

        let hnsw: Hnsw<SillyComparator> = Hnsw::generate(c, vs, 3, 6, 6);
        hnsw
    }

    fn make_broken_hnsw() -> Hnsw<SillyComparator> {
        let sqrt2_recip = std::f32::consts::FRAC_1_SQRT_2;
        let data: Vec<SillyVec> = vec![
            [1.0, 0.0, 0.0],                 // 0
            [0.0, 1.0, 0.0],                 // 1
            [0.0, 0.0, 1.0],                 // 2
            [sqrt2_recip, sqrt2_recip, 0.0], // 3
            [0.5773, 0.5773, 0.5773],        // 4
            [-1.0, 0.0, 0.0],                // 5
            [0.0, -1.0, 0.0],                // 6
            [0.0, 0.0, -1.0],                // 7
            [0.0, sqrt2_recip, sqrt2_recip], // 8
            [sqrt2_recip, 0.0, sqrt2_recip], // 9
        ];
        let c = SillyComparator { data: data.clone() };
        let vs: Vec<_> = (0..9).map(VectorId).collect(); // only index 8 first..

        let mut hnsw: Hnsw<SillyComparator> = Hnsw::generate(c, vs, 3, 6, 6);
        let bottom = &mut hnsw.layers[1];
        // add a ninth disconnected vector
        bottom.nodes.push(VectorId(9));
        bottom.neighbors.extend(vec![NodeId(!0); 6]);
        hnsw
    }

    #[test]
    fn test_nearness_search() {
        let hnsw: Hnsw<SillyComparator> = make_simple_hnsw();
        let sqrt2_recip = std::f32::consts::FRAC_1_SQRT_2;
        let slice = &[0.0, sqrt2_recip, sqrt2_recip];
        let search_vector = AbstractVector::Unstored(slice);
        let results = hnsw.search(search_vector, 9, 1);
        assert_eq!(
            results,
            vec![
                (VectorId(8), 5.9604645e-8),
                (VectorId(4), 0.1835745),
                (VectorId(1), 0.29289323),
                (VectorId(2), 0.29289323),
                (VectorId(3), 0.5),
                (VectorId(0), 1.0),
                (VectorId(5), 1.0),
                (VectorId(6), 1.7071068),
                (VectorId(7), 1.7071068)
            ],
        )
    }

    #[test]
    fn test_generation() {
        let hnsw: Hnsw<SillyComparator> = make_simple_hnsw();
        assert_eq!(
            hnsw.get_layer(0).map(|layer| &layer.nodes),
            Some(
                vec![
                    VectorId(0),
                    VectorId(1),
                    VectorId(2),
                    VectorId(3),
                    VectorId(4),
                    VectorId(5),
                    VectorId(6),
                    VectorId(7),
                    VectorId(8)
                ]
                .as_ref()
            )
        );
        assert_eq!(
            hnsw.get_layer(0).map(|layer| &layer.neighbors),
            Some(
                vec![
                    NodeId(3),
                    NodeId(4),
                    NodeId(1),
                    NodeId(2),
                    NodeId(6),
                    NodeId(7),
                    NodeId(3),
                    NodeId(8),
                    NodeId(4),
                    NodeId(0),
                    NodeId(2),
                    NodeId(5),
                    NodeId(8),
                    NodeId(4),
                    NodeId(0),
                    NodeId(1),
                    NodeId(3),
                    NodeId(5),
                    NodeId(4),
                    NodeId(0),
                    NodeId(1),
                    NodeId(8),
                    NodeId(2),
                    NodeId(7),
                    NodeId(3),
                    NodeId(8),
                    NodeId(0),
                    NodeId(1),
                    NodeId(2),
                    NodeId(5),
                    NodeId(1),
                    NodeId(2),
                    NodeId(6),
                    NodeId(8),
                    NodeId(4),
                    NodeId(3),
                    NodeId(0),
                    NodeId(2),
                    NodeId(5),
                    NodeId(7),
                    NodeId(4),
                    NodeId(3),
                    NodeId(0),
                    NodeId(1),
                    NodeId(3),
                    NodeId(6),
                    NodeId(4),
                    NodeId(8),
                    NodeId(4),
                    NodeId(1),
                    NodeId(2),
                    NodeId(3),
                    NodeId(0),
                    NodeId(5)
                ]
                .as_ref()
            )
        );
    }

    #[test]
    fn test_search() {
        let hnsw: Hnsw<SillyComparator> = make_simple_hnsw();
        let data = &hnsw.layers[0].comparator.data;
        for (i, datum) in data.iter().enumerate() {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 9, 1);
            eprintln!("results: {results:?}");
            assert_eq!(VectorId(i), results[0].0)
        }
    }

    fn do_test_recall(hnsw: &Hnsw<BigComparator>, minimum_recall: f32) -> f32 {
        let data = &hnsw.layers[0].comparator.data;
        let total = data.len();
        let total_relevant: usize = data
            .par_iter()
            .enumerate()
            .map(|(i, datum)| {
                /* eprintln!("XXXXXXXXXXXXXXXXXXXXXX");
                eprintln!("Searching for {i}");
                */
                let v = AbstractVector::Unstored(datum);
                let results = hnsw.search(v, 300, 2);
                if VectorId(i) == results[0].0 {
                    1
                } else {
                    0
                }
            })
            .sum();
        eprintln!("total relevant: {total_relevant}");
        eprintln!("from total: {total}");
        let recall = total_relevant as f32 / total as f32;
        eprintln!("with recall: {recall}");
        assert!(recall >= minimum_recall);

        recall
    }

    #[test]
    fn test_supers() {
        let size = 10000;
        let dimension = 10;
        let hnsw: Hnsw<BigComparator> = bigvec::make_random_hnsw(size, dimension);
        //eprintln!("Top neighbors: {:?}", hnsw.layers[0].neighbors);
        let supers_1 = hnsw.supers_for_layer(0);
        let supers_2 = hnsw.supers_for_layer(0);
        assert_eq!(supers_1, supers_2);
        let layer = hnsw.get_layer(0).unwrap();
        let node_distances1 = layer.node_distances(supers_1);
        let node_distances2 = layer.node_distances(supers_1);
        assert_eq!(node_distances1, node_distances2);
        let n1 = layer.discover_nodes_to_promote(supers_1);
        let n2 = layer.discover_nodes_to_promote(supers_1);
        assert_eq!(n1, n2);
        let v1 = hnsw.discover_vectors_to_promote(0);
        eprintln!("{v1:?}");
        let v2 = hnsw.discover_vectors_to_promote(0);
        assert_eq!(v1, v2);
        panic!();
    }

    #[test]
    fn test_recall() {
        let size = 10_000;
        let dimension = 1536;
        let mut hnsw: Hnsw<BigComparator> = bigvec::make_random_hnsw(size, dimension);
        hnsw.improve_index(0.01, 0.01, 1.0, None);
        do_test_recall(&hnsw, 1.0);
        panic!();
    }

    #[test]
    fn test_promotion() {
        let size = 1000;
        let dimension = 50;
        let mut hnsw: Hnsw<BigComparator> =
            bigvec::make_random_hnsw_with_order(size, dimension, 10);
        hnsw.improve_index(0.01, 0.01, 1.0, None);
        do_test_recall(&hnsw, 0.0);
        let mut improvement_count = 0;
        let mut last_recall = 0.0;
        let mut last_improvement = 1.0;
        while last_improvement > 0.001 {
            eprintln!("{improvement_count} time to improve index");
            hnsw.improve_index(0.01, 0.01, 1.0, None);
            let new_recall = do_test_recall(&hnsw, 0.0);
            last_improvement = new_recall - last_recall;
            last_recall = new_recall;
            eprintln!("improved index by {last_improvement}");
            improvement_count += 1;
            eprintln!("=========");
        }
        panic!();
    }

    #[test]
    fn test_small_index_improvement() {
        let mut hnsw: Hnsw<SillyComparator> = make_simple_hnsw();
        eprintln!("One from bottom: {:?}", hnsw.layers[hnsw.layer_count() - 2]);
        hnsw.improve_index(0.1, 0.1, 1.0, None);
        eprintln!(
            "One from bottom after: {:?}",
            hnsw.layers[hnsw.layer_count() - 2]
        );
        let data = &hnsw.layers[hnsw.layer_count() - 1].comparator.data;
        for (i, datum) in data.iter().enumerate() {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 9, 1);
            assert_eq!(VectorId(i), results[0].0)
        }
    }

    #[test]
    fn test_tiny_index_improvement() {
        let mut hnsw: Hnsw<SillyComparator> = make_broken_hnsw();
        hnsw.improve_index(0.1, 0.1, 1.0, None);
        let data = &hnsw.layers[hnsw.layer_count() - 1].comparator.data;
        for (i, datum) in data.iter().enumerate() {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 9, 1);
            assert_eq!(VectorId(i), results[0].0)
        }
    }

    #[test]
    fn test_partitions_with_single_entry() {
        let partitions = calculate_partitions(1, 24);
        assert_eq!(1, partitions.len());
    }

    #[test]
    fn test_neighborhood_order() {
        let size = 10_000;
        let dimension = 1536;
        let orders = vec![6, 12, 24];
        let mut best = 0.0_f32;
        let mut best_order = usize::MAX;
        let mut best_hnsw = None;
        for order in orders {
            let hnsw: Hnsw<BigComparator> =
                bigvec::make_random_hnsw_with_order(size, dimension, order);
            let recall = do_test_recall(&hnsw, 0.0);
            if recall > best {
                best = recall;
                best_hnsw = Some(hnsw);
                best_order = order
            }
        }
        eprintln!("best_order: {best_order}");
        let mut improvement_count = 0;
        let mut last_recall = best;
        let mut last_improvement = 1.0;
        let mut hnsw = best_hnsw.unwrap();
        while last_improvement > 0.001 {
            eprintln!("{improvement_count} time to improve index");
            hnsw.improve_index(0.1, 0.1, 1.0, None);
            let new_recall = do_test_recall(&hnsw, 0.0);
            last_improvement = new_recall - last_recall;
            last_recall = new_recall;
            eprintln!("improved index by {last_improvement}");
            improvement_count += 1;
            eprintln!("=========");
        }

        panic!();
    }

    #[test]
    fn calculate_partitions_with_additions() {
        let order = 2;
        let mut sizes = calculate_partitions(1000, order);
        sizes.reverse();
        eprintln!("sizes: {:?}", sizes);
        let promotion_count = 100;
        let result = calculate_partitions_for_additions(&sizes[1..], promotion_count, order);
        let expected = vec![100, 50, 25, 13, 6, 3, 2, 1, 1, 1];
        assert_eq!(expected, result);
        panic!();
    }

    #[test]
    fn test_knn() {
        let hnsw: Hnsw<SillyComparator> = make_simple_hnsw();
        let mut results: Vec<_> = hnsw.knn(1, 1).collect();
        results.sort_by_key(|(v, _d)| *v);
        assert_eq!(
            results,
            vec![
                (VectorId(0), vec![(VectorId(3), 0.29289323)]),
                (VectorId(1), vec![(VectorId(3), 0.29289323)]),
                (VectorId(2), vec![(VectorId(8), 0.29289323)]),
                (VectorId(3), vec![(VectorId(4), 0.1835745)]),
                (VectorId(4), vec![(VectorId(3), 0.1835745)]),
                (VectorId(5), vec![(VectorId(1), 1.0)]),
                (VectorId(6), vec![(VectorId(0), 1.0)]),
                (VectorId(7), vec![(VectorId(0), 1.0)]),
                (VectorId(8), vec![(VectorId(4), 0.1835745)])
            ]
        );
    }

    #[test]
    fn test_threshold_nn() {
        let hnsw: Hnsw<SillyComparator> = make_simple_hnsw();
        let mut results: Vec<_> = hnsw
            .threshold_nn(0.3, 1, hnsw.zero_layer_neighborhood_size)
            .collect();
        results.sort_by_key(|(v, _d)| *v);
        assert_eq!(
            results,
            vec![
                (VectorId(0), vec![(VectorId(3), 0.29289323)]),
                (
                    VectorId(1),
                    vec![(VectorId(3), 0.29289323), (VectorId(8), 0.29289323)],
                ),
                (VectorId(2), vec![(VectorId(8), 0.29289323)]),
                (
                    VectorId(3),
                    vec![
                        (VectorId(4), 0.1835745),
                        (VectorId(0), 0.29289323),
                        (VectorId(1), 0.29289323),
                    ],
                ),
                (
                    VectorId(4),
                    vec![(VectorId(3), 0.1835745), (VectorId(8), 0.1835745)],
                ),
                (VectorId(5), vec![]),
                (VectorId(6), vec![]),
                (VectorId(7), vec![]),
                (
                    VectorId(8),
                    vec![
                        (VectorId(4), 0.1835745),
                        (VectorId(1), 0.29289323),
                        (VectorId(2), 0.29289323),
                    ],
                ),
            ]
        );
    }

    pub type Vec32 = Vec<f32>;
    #[derive(Clone, Debug, PartialEq)]
    pub struct Comparator32 {
        pub data: Arc<Vec<Vec32>>,
    }

    impl Comparator for Comparator32 {
        type T = BigVec;
        type Borrowable<'a> = &'a BigVec;
        fn compare_raw(&self, v1: &BigVec, v2: &BigVec) -> f32 {
            let mut result = 0.0;
            for (&f1, &f2) in v1.iter().zip(v2.iter()) {
                result += (f1 - f2).powi(2)
            }
            result.powf(0.5)
        }
        fn lookup(&self, v: VectorId) -> &BigVec {
            &self.data[v.0]
        }
    }

    pub fn random_vec(prng: &mut StdRng, size: usize) -> Vec<f32> {
        let range = Uniform::from(-1.0..1.0);
        let vec: Vec<f32> = prng.sample_iter(&range).take(size).collect();
        vec
    }

    #[test]
    fn test_euclidean() {
        let mut prng = StdRng::seed_from_u64(42);
        let vecs: Vec<Vec<f32>> = (0..10_000)
            .map(move |_| random_vec(&mut prng, 32))
            .collect();
        let cc = Comparator32 { data: vecs.into() };
        let vids: Vec<VectorId> = (0..10_000).map(VectorId).collect();
        let mut hnsw: Hnsw<Comparator32> = Hnsw::generate(cc, vids, 24, 48, 12);
        hnsw.improve_neighbors(0.01, 1.0, None);
        panic!()
    }
}
