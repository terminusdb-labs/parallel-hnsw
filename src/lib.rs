use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io,
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
    slice::{self, Iter},
    sync::{
        atomic::{self, AtomicUsize},
        RwLock,
    },
};

use thiserror::Error;

use itertools::Itertools;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::io::{Read, Write};

use crate::priority_queue::{EmptyValue, PriorityQueue};

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct VectorId(pub usize);
#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct NodeId(pub usize);

impl VectorId {
    const MAX: Self = Self(!0);
}
impl NodeId {
    const MAX: Self = Self(!0);
}

impl EmptyValue for NodeId {
    fn is_empty(&self) -> bool {
        self.0 == !0
    }

    fn empty() -> Self {
        Self::MAX
    }
}

impl EmptyValue for VectorId {
    fn is_empty(&self) -> bool {
        self.0 == !0
    }

    fn empty() -> Self {
        Self::MAX
    }
}

mod priority_queue;

pub enum AbstractVector<'a, T> {
    Stored(VectorId),
    Unstored(&'a T),
}

impl<'a, T> Debug for AbstractVector<'a, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stored(arg0) => f.debug_tuple("Stored").field(arg0).finish(),
            Self::Unstored(arg0) => f.debug_tuple("Unstored").field(arg0).finish(),
        }
    }
}

impl<'a, T> Clone for AbstractVector<'a, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Stored(arg0) => Self::Stored(*arg0),
            Self::Unstored(arg0) => Self::Unstored(arg0),
        }
    }
}

#[derive(Error, Debug)]
pub enum SerializationError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

pub trait Comparator<T>: Sync + Clone {
    type Params;
    fn compare_vec(&self, v1: AbstractVector<T>, v2: AbstractVector<T>) -> f32;
    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError>;
    fn deserialize<P: AsRef<Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, SerializationError>;
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedFloat(pub f32);

impl Eq for OrderedFloat {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Layer<C: Comparator<T>, T> {
    pub comparator: C,
    pub neighborhood_size: usize,
    pub nodes: Vec<VectorId>,
    pub neighbors: Vec<NodeId>,
    _phantom: PhantomData<T>,
}

impl<C: Comparator<T>, T> Clone for Layer<C, T> {
    fn clone(&self) -> Self {
        Self {
            comparator: self.comparator.clone(),
            neighborhood_size: self.neighborhood_size,
            nodes: self.nodes.clone(),
            neighbors: self.neighbors.clone(),
            _phantom: PhantomData,
        }
    }
}

unsafe impl<C: Comparator<T>, T> Sync for Layer<C, T> {}

impl<C: Comparator<T>, T> AsRef<Layer<C, T>> for Layer<C, T> {
    fn as_ref(&self) -> &Layer<C, T> {
        self
    }
}

type NodeDistances = Vec<(NodeId, f32)>;

impl<C: Comparator<T>, T> Layer<C, T> {
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
        v: AbstractVector<T>,
        candidates: &mut PriorityQueue<NodeId>,
        mut probe_depth: usize,
    ) {
        assert!(!candidates.is_empty());
        let mut visit_queue: Vec<(NodeId, f32)> = candidates.iter().collect();
        visit_queue.reverse();
        let mut visited: HashSet<NodeId> = candidates.iter().map(|(n, _)| n).collect();
        //eprintln!("------------------------------------");
        //eprintln!("Initial visit queue: {visit_queue:?}");
        while let Some((next, _)) = visit_queue.pop() {
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

            visit_queue.extend(neighbor_distances.iter());
            //eprintln!("before");
            //dbg!(&candidates.data);
            //dbg!(&candidates.priorities);

            let did_something = candidates.merge_pairs(&neighbor_distances);
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
            visit_queue.sort_by_key(|(n, distance)| (OrderedFloat(-*distance), *n))
        }
    }

    pub fn closest_vectors(
        &self,
        v: AbstractVector<T>,
        candidates: &PriorityQueue<VectorId>,
        candidate_count: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        let pairs: Vec<(NodeId, f32)> = candidates
            .iter()
            // We should only be proceeding downwards!
            .map(|(v, d)| (self.get_node(v).unwrap(), d))
            .collect();
        //eprintln!("pairs: {pairs:?}");
        let mut queue = PriorityQueue::new(candidate_count);
        queue.merge_pairs(&pairs);
        self.closest_nodes(v, &mut queue, probe_depth);
        queue
            .iter()
            .map(|(node_id, distance)| (self.get_vector(node_id), distance))
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
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
            .take_while(|(_, _, d)| *d > 48)
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
    const MAX: Self = Self {
        hops: usize::MAX,
        index_sum: usize::MAX,
    };
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct HnswSearcher<C: Comparator<T>, T> {
    _phantom: PhantomData<(C, T)>,
}

impl<C: Comparator<T>, T: Sync> Default for HnswSearcher<C, T> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Hnsw<C: Comparator<T>, T: Sync> {
    pub layers: Vec<Layer<C, T>>,
    immutable: HnswSearcher<C, T>,
}

impl<C: Comparator<T>, T: Sync> HnswSearcher<C, T> {
    pub fn compare_all(comparator: C, v: VectorId, vs: &[VectorId]) -> Vec<(VectorId, f32)> {
        let mut res: Vec<_> = vs
            .iter()
            .filter(|w| **w != v)
            .map(|w| {
                (
                    *w,
                    comparator.compare_vec(AbstractVector::Stored(v), AbstractVector::Stored(*w)),
                )
            })
            .collect();
        res.sort_by_key(|(v, d)| (OrderedFloat(*d), *v));
        res
    }

    pub fn entry_vector(&self) -> VectorId {
        // Other choices are possible
        VectorId(0)
    }

    fn generate_initial_partitions<L: AsRef<Layer<C, T>> + Sync>(
        &self,
        vs: &[VectorId],
        comparator: &C,
        number_of_supers_to_check: usize,
        layers: &[L],
    ) -> Vec<(NodeId, VectorId, NodeDistances)> {
        let mut initial_partitions: Vec<(NodeId, VectorId, NodeDistances)> =
            Vec::with_capacity(vs.len());
        vs.par_iter()
            .enumerate()
            .map(|(node_id, vector_id)| {
                let comparator = comparator.clone();
                let initial_vector_distances = if layers.is_empty() {
                    //eprintln!("empty layers");
                    HnswSearcher::compare_all(comparator, *vector_id, vs)
                } else {
                    //eprintln!("not empty layers");
                    self.initial_vector_distances(*vector_id, number_of_supers_to_check, layers)
                };
                //eprintln!("ivd: {initial_vector_distances:?}");
                let initial_node_distances: Vec<_> = initial_vector_distances
                    .into_iter()
                    .map(|(vector_id, distance)| {
                        (NodeId(vs.binary_search(&vector_id).unwrap()), distance)
                    })
                    .collect();
                (NodeId(node_id), *vector_id, initial_node_distances)
            })
            .collect_into_vec(&mut initial_partitions);

        initial_partitions.par_sort_unstable_by_key(|(_node_id, _vector_id, distances)| {
            distances.first().map(|(_, d)| OrderedFloat(*d))
        });
        initial_partitions
    }

    pub fn initial_vector_distances<L: AsRef<Layer<C, T>>>(
        &self,
        v: VectorId,
        number_of_nodes: usize,
        layers: &[L],
    ) -> Vec<(VectorId, f32)> {
        self.search_layers(AbstractVector::Stored(v), number_of_nodes, layers, 1)
            .into_iter()
            .filter(|(w, _)| v != *w)
            .collect::<Vec<_>>()
    }

    pub fn search_layers<L: AsRef<Layer<C, T>>>(
        &self,
        v: AbstractVector<T>,
        number_of_candidates: usize,
        layers: &[L],
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        self.search_layers_noisy(v, number_of_candidates, layers, probe_depth, false)
    }

    pub fn search_layers_noisy<L: AsRef<Layer<C, T>>>(
        &self,
        v: AbstractVector<T>,
        number_of_candidates: usize,
        layers: &[L],
        probe_depth: usize,
        noisy: bool,
    ) -> Vec<(VectorId, f32)> {
        let upper_layer_candidate_count = 1;
        let entry_vector = self.entry_vector();
        let distance_from_entry = layers
            .first()
            .map(|l| {
                l.as_ref()
                    .comparator
                    .compare_vec(v.clone(), AbstractVector::Stored(entry_vector))
            })
            .unwrap_or(0.0);
        if noisy {
            eprintln!("layer len: {}", layers.len());
            eprintln!("distance from entry: {distance_from_entry}");
        }
        let mut candidates = PriorityQueue::new(number_of_candidates);
        candidates.insert(entry_vector, distance_from_entry);
        for i in 0..layers.len() {
            candidates
                .iter()
                .fold(f32::MIN, |last, (nodeid, distance)| {
                    if distance < last {
                        panic!("oh yikes {nodeid:?} {distance}");
                    }
                    distance
                });
            let candidate_count = if layers.len() == 1 || i == layers.len() - 1 {
                number_of_candidates
            } else {
                upper_layer_candidate_count
            };
            let layer = &layers[i];
            let closest = layer.as_ref().closest_vectors(
                v.clone(),
                &candidates,
                candidate_count,
                probe_depth,
            );
            if noisy {
                eprintln!("closest: {closest:?}");
            }
            candidates.merge_pairs(&closest);
        }

        candidates.iter().collect()
    }
}

impl<C: Comparator<T> + 'static, T: Sync + 'static> Hnsw<C, T> {
    pub fn vector_count(&self) -> usize {
        self.get_layer(0).map(|l| l.node_count()).unwrap_or(0)
    }

    pub fn get_layer(&self, i: usize) -> Option<&Layer<C, T>> {
        self.get_layer_from_top(self.layers.len() - i - 1)
    }

    pub fn get_layer_from_top_mut(&mut self, i: usize) -> Option<&mut Layer<C, T>> {
        if i < self.layer_count() {
            Some(&mut self.layers[i])
        } else {
            // eprintln!("No layer");
            None
        }
    }

    pub fn get_layer_from_top(&self, i: usize) -> Option<&Layer<C, T>> {
        if i < self.layer_count() {
            Some(&self.layers[i])
        } else {
            // eprintln!("No layer");
            None
        }
    }

    pub fn get_layer_above(&self, i: usize) -> Option<&Layer<C, T>> {
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

    pub fn search(
        &self,
        v: AbstractVector<T>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        self.immutable
            .search_layers(v, number_of_candidates, &self.layers, probe_depth)
    }

    pub fn search_noisy(
        &self,
        v: AbstractVector<T>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        self.immutable
            .search_layers_noisy(v, number_of_candidates, &self.layers, probe_depth, true)
    }

    pub fn generate_layer(
        &self,
        comparator: C,
        vs: Vec<VectorId>,
        neighborhood_size: usize,
    ) -> Layer<C, T> {
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
        let initial_partitions = self.immutable.generate_initial_partitions(
            &vs,
            &comparator,
            number_of_supers_to_check,
            &self.layers,
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
            _phantom: PhantomData,
        }
    }

    pub fn generate(
        c: C,
        vs: Vec<VectorId>,
        neighborhood_size: usize,
        zero_layer_neighborhood_size: usize,
    ) -> Self {
        let total_size = vs.len();
        // eprintln!("neighborhood_size: {neighborhood_size}");
        // eprintln!("total_size: {total_size}");
        // eprintln!("layer count: {layer_count}");
        let partitions = calculate_partitions(total_size, neighborhood_size);
        let layer_count = partitions.len();
        let layers = Vec::with_capacity(layer_count);
        let mut hnsw: Hnsw<C, T> = Hnsw {
            layers,
            immutable: HnswSearcher::default(),
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
            let layer = hnsw.generate_layer(c.clone(), slice.to_vec(), neighbors);
            hnsw.layers.push(layer)
        }

        hnsw
    }

    pub fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        std::fs::create_dir_all(&path)?;
        let mut hnsw_meta: PathBuf = path.as_ref().into();
        hnsw_meta.push("meta");
        eprintln!("hnsw serialization path: {hnsw_meta:?}");
        let mut hnsw_meta_file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(hnsw_meta)?;
        eprintln!("opened hnsw file");
        let layer_count = self.layer_count();

        let serialized = serde_json::to_string(&HNSWMeta { layer_count })?;
        eprintln!("serialized data");
        hnsw_meta_file.write_all(serialized.as_bytes())?;
        eprintln!("serialized to file");

        if layer_count > 0 {
            let mut hnsw_comparator: PathBuf = path.as_ref().into();
            hnsw_comparator.push("comparator");
            self.layers[0].comparator.serialize(hnsw_comparator)?;
        }

        let layer_count = self.layer_count();

        for i in 0..layer_count {
            let layer = &self.layers[i];
            let layer_number = layer_count - i - 1;

            // Write meta data
            let mut hnsw_layer_meta: PathBuf = path.as_ref().into();
            hnsw_layer_meta.push(format!("layer.meta.{layer_number}"));
            let mut hnsw_layer_meta_file: std::fs::File = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&hnsw_layer_meta)?;
            eprintln!("opened {hnsw_layer_meta:?} for layer {layer_number}");
            let neighborhood_size = layer.neighborhood_size;
            let node_count = layer.nodes.len();
            let layer_meta = serde_json::to_string(&LayerMeta {
                node_count,
                neighborhood_size,
            })?;
            hnsw_layer_meta_file.write_all(&layer_meta.into_bytes())?;
            eprintln!("wrote meta for layer {layer_number}");

            // Write Nodes
            let mut hnsw_layer_nodes: PathBuf = path.as_ref().into();
            hnsw_layer_nodes.push(format!("layer.nodes.{layer_number}"));
            let mut hnsw_layer_nodes_file: std::fs::File = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&hnsw_layer_nodes)?;
            eprintln!("opened {hnsw_layer_nodes:?} for layer {layer_number}");
            let node_slice_u8: &[u8] = unsafe {
                let nodes: &[VectorId] = &layer.nodes;
                let ptr = nodes.as_ptr() as *const u8;
                let size = layer.nodes.len() * mem::size_of::<VectorId>();
                slice::from_raw_parts(ptr, size)
            };
            hnsw_layer_nodes_file.write_all(node_slice_u8)?;
            eprintln!("wrote nodes for layer {layer_number}");

            // Write Neighbors
            let mut hnsw_layer_neighbors: PathBuf = path.as_ref().into();
            hnsw_layer_neighbors.push(format!("layer.neighbors.{layer_number}"));
            let mut hnsw_layer_neighbors_file = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&hnsw_layer_neighbors)?;
            eprintln!("opened {hnsw_layer_neighbors_file:?} for layer {layer_number}");
            let neighbor_slice_u8: &[u8] = unsafe {
                let neighbors: &[NodeId] = &layer.neighbors;
                let ptr = neighbors.as_ptr() as *const u8;
                let size = layer.neighbors.len() * mem::size_of::<NodeId>();
                slice::from_raw_parts(ptr, size)
            };
            hnsw_layer_neighbors_file.write_all(neighbor_slice_u8)?;
            eprintln!("wrote neighbors for layer {layer_number}");
        }
        Ok(())
    }

    pub fn deserialize<P: AsRef<Path>>(
        path: P,
        params: C::Params,
    ) -> Result<Option<Self>, SerializationError> {
        let mut hnsw_meta: PathBuf = path.as_ref().into();
        hnsw_meta.push("meta");
        let mut hnsw_meta_file = OpenOptions::new().read(true).open(dbg!(hnsw_meta))?;
        let mut contents = String::new();
        hnsw_meta_file.read_to_string(&mut contents)?;
        let HNSWMeta { layer_count }: HNSWMeta = serde_json::from_str(&contents)?;

        let mut hnsw_comparator_path: PathBuf = dbg!(path.as_ref().into());
        hnsw_comparator_path.push("comparator");

        // If we don't have a comparator, the HNSW is empty
        if hnsw_comparator_path.exists() {
            let comparator: C = Comparator::deserialize(&hnsw_comparator_path, params)?;
            let mut layers = Vec::with_capacity(layer_count);
            for i in 0..layer_count {
                let layer_number = layer_count - i - 1;
                // Read meta database_
                let mut hnsw_layer_meta: PathBuf = path.as_ref().into();
                hnsw_layer_meta.push(format!("layer.meta.{layer_number}"));
                let mut hnsw_layer_meta_file: std::fs::File =
                    OpenOptions::new().read(true).open(dbg!(hnsw_layer_meta))?;
                let mut contents = String::new();
                hnsw_layer_meta_file.read_to_string(&mut contents)?;
                let LayerMeta {
                    node_count,
                    neighborhood_size,
                } = serde_json::from_str(&contents)?;

                let mut hnsw_layer_nodes: PathBuf = path.as_ref().into();
                hnsw_layer_nodes.push(format!("layer.nodes.{layer_number}"));
                let mut hnsw_layer_nodes_file: std::fs::File =
                    OpenOptions::new().read(true).open(dbg!(hnsw_layer_nodes))?;
                let mut nodes: Vec<VectorId> = Vec::with_capacity(node_count);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    nodes.set_len(node_count);
                }
                let size = node_count * mem::size_of::<VectorId>();
                let node_slice_u8: &mut [u8] = unsafe {
                    let nodes: &mut [VectorId] = &mut nodes;
                    let ptr = nodes.as_mut_ptr() as *mut u8;
                    slice::from_raw_parts_mut(ptr, size)
                };
                hnsw_layer_nodes_file.read_exact(node_slice_u8)?;

                let mut hnsw_layer_neighbors: PathBuf = path.as_ref().into();
                hnsw_layer_neighbors.push(format!("layer.neighbors.{layer_number}"));
                let mut hnsw_layer_neighbors_file: std::fs::File = OpenOptions::new()
                    .read(true)
                    .open(dbg!(hnsw_layer_neighbors))?;
                let mut neighbors: Vec<NodeId> = Vec::with_capacity(node_count * neighborhood_size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    neighbors.set_len(node_count * neighborhood_size);
                }
                let neighbor_slice_u8: &mut [u8] = unsafe {
                    let neighbors: &mut [NodeId] = &mut neighbors;
                    let ptr = neighbors.as_mut_ptr() as *mut u8;
                    let size = (node_count * neighborhood_size) * mem::size_of::<NodeId>();
                    slice::from_raw_parts_mut(ptr, size)
                };
                hnsw_layer_neighbors_file.read_exact(neighbor_slice_u8)?;

                layers.push(Layer {
                    comparator: comparator.clone(),
                    neighborhood_size,
                    neighbors,
                    nodes,
                    _phantom: PhantomData,
                });
            }
            Ok(Some(Hnsw {
                layers,
                immutable: HnswSearcher::default(),
            }))
        } else {
            Ok(None)
        }
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

    pub fn discover_vectors_to_promote(&self, layer_id: usize) -> Vec<VectorId> {
        // We need to start with layer zero and proceed upwards
        self.get_layer(layer_id)
            .map(|layer| {
                let supers = self.supers_for_layer(layer_id);
                layer
                    .discover_nodes_to_promote(supers)
                    .iter()
                    .map(|n| layer.get_vector(*n))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    pub fn extend_layer(&mut self, layer_id: usize, vecs: Vec<VectorId>) {
        let layer_id_from_top = self.layer_count() - layer_id - 1;
        eprintln!("Extending layer: {layer_id:?}");
        eprintln!("Counting as {layer_id_from_top:?}");
        let searcher = &self.immutable;
        let (layers_above, layers_below) = self.layers.split_at_mut(layer_id_from_top);
        eprintln!(
            "layers above: {}, layers_below: {}",
            layers_above.len(),
            layers_below.len()
        );
        let layers_above: &[Layer<C, T>] = layers_above;
        let layer = &mut layers_below[0];

        let (old_nodes, old_nodes_map, new_nodes_map, vecs) = generate_node_maps(vecs, layer);

        assert_eq!(old_nodes_map.len() + new_nodes_map.len(), layer.nodes.len());

        let new_neighbors_len = layer.nodes.len() * layer.neighborhood_size;
        let mut old_neighbors = Vec::with_capacity(new_neighbors_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            old_neighbors.set_len(new_neighbors_len)
        };
        std::mem::swap(&mut layer.neighbors, &mut old_neighbors);

        copy_old_neighbhoods_into_layer(&old_nodes, &old_neighbors, &old_nodes_map, layer);

        let borrowed_comparator = &layer.comparator;
        let cross_compare = cross_compare_vectors(&vecs, borrowed_comparator);

        let mut pseudo_layers: Vec<&Layer<_, _>> = Vec::new();
        pseudo_layers.extend(layers_above.iter());
        let pseudo_layer = Layer {
            comparator: layer.comparator.clone(),
            neighborhood_size: layer.neighborhood_size,
            nodes: old_nodes,
            neighbors: old_neighbors,
            _phantom: PhantomData,
        };
        pseudo_layers.push(&pseudo_layer);
        eprintln!("Finding neighborhoods");
        let mut neighborhood_candidates: Vec<(NodeId, NodeId, f32)> = vecs
            .par_iter()
            .flat_map(|v| {
                let mut distances: Vec<(VectorId, f32)> = searcher.search_layers(
                    AbstractVector::Stored(*v),
                    10 * layer.neighborhood_size,
                    &pseudo_layers,
                    1,
                );
                let cross_results = cross_compare.get(v).unwrap();

                distances.extend(cross_results);

                distances.sort_by_key(|(v, d)| (*v, OrderedFloat(*d)));
                distances.dedup_by_key(|(v, _)| *v);
                distances.sort_by_key(|(v, d)| (OrderedFloat(*d), *v));
                distances.truncate(layer.neighborhood_size);

                let neighborhood_distances: Vec<(NodeId, f32)> = distances
                    .into_iter()
                    .map(|(w, d)| (layer.get_node(w).unwrap(), d))
                    .collect();

                let mut neighborhood: Vec<NodeId> =
                    neighborhood_distances.iter().map(|(n, _)| *n).collect();
                neighborhood.resize(layer.neighborhood_size, NodeId(!0));

                // Write neighborhood directly here!
                let our_node = layer.get_node(*v).unwrap();
                let new_neighborhood = unsafe {
                    std::slice::from_raw_parts_mut(
                        layer
                            .neighbors
                            .as_ptr()
                            .add(our_node.0 * layer.neighborhood_size)
                            as *mut NodeId,
                        layer.neighborhood_size,
                    )
                };
                new_neighborhood.copy_from_slice(&neighborhood);

                neighborhood_distances
                    .into_par_iter()
                    .map(move |(w, d)| (w, our_node, d))
            })
            .collect();
        eprintln!("Grouping neighborhoods");
        neighborhood_candidates.par_sort_by_key(|(n1, n2, d)| (*n1, *n2, OrderedFloat(*d)));
        let final_groups = neighborhood_candidates
            .into_iter()
            .into_group_map_by(|(n, _, _)| *n);

        final_groups.into_par_iter().for_each(|(node, group)| {
            // 1. find all distances to existing neighbors for this node
            let neighborhood = unsafe {
                std::slice::from_raw_parts_mut(
                    layer
                        .neighbors
                        .as_ptr()
                        .add(node.0 * layer.neighborhood_size) as *mut NodeId,
                    layer.neighborhood_size,
                )
            };
            let mut distances: NodeDistances = Vec::new();
            for neighbor in neighborhood.iter().take_while(|n| n.0 != !0) {
                let distance = layer.comparator.compare_vec(
                    AbstractVector::Stored(layer.get_vector(node)),
                    AbstractVector::Stored(layer.get_vector(*neighbor)),
                );

                distances.push((*neighbor, distance));
            }
            // 2. add the new distances that we got in 'group'
            for (_, neighbor, distance) in group {
                distances.push((neighbor, distance));
            }
            // 3. sort, truncate.
            distances.sort_by_key(|(n, distance)| (OrderedFloat(*distance), *n));
            distances.dedup();
            distances.truncate(layer.neighborhood_size);

            // 4. write back new neighbor list.
            let mut new_neighborhood: Vec<_> = distances.into_iter().map(|(n, _)| n).collect();
            new_neighborhood.resize(layer.neighborhood_size, NodeId(!0));

            neighborhood.copy_from_slice(&new_neighborhood)
        });
    }

    pub fn link_layer_to_better_neighbors(&mut self, layer_from_top: usize) -> usize {
        let (top_stack, bottom_stack) = self.layers.split_at_mut(layer_from_top);
        let current_layer = &mut bottom_stack[0];

        let pseudo_layer = current_layer.clone();
        let mut pseudo_stack: Vec<&Layer<C, T>> = Vec::with_capacity(top_stack.len() + 1);
        pseudo_stack.extend(top_stack.iter());
        pseudo_stack.push(&pseudo_layer);

        (0..current_layer.node_count())
            .into_par_iter()
            .map(|node_id| {
                let local_node = NodeId(node_id);
                let vector = current_layer.get_vector(local_node);
                let matches = self.immutable.search_layers(
                    AbstractVector::Stored(vector),
                    100,
                    &pseudo_stack,
                    2,
                );
                for (neighbor_vec, distance) in matches.into_iter().take(10) {
                    if neighbor_vec == vector {
                        break;
                    }
                    let neighbor = current_layer.get_node(neighbor_vec).unwrap();
                    let neighbor_of_neighbors = current_layer.get_neighbors(neighbor);
                    if let Some(insert_pos) = neighbor_of_neighbors.iter().position(|n| {
                        if n.is_empty() || *n == local_node {
                            return true;
                        }
                        let v = current_layer.get_vector(*n);
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

                        let neighbor_of_neighbors: &'static mut [NodeId] = unsafe {
                            slice::from_raw_parts_mut(
                                neighbor_of_neighbors.as_ptr() as *mut NodeId,
                                neighbor_of_neighbors.len(),
                            )
                        };
                        for i in (insert_pos..neighbor_of_neighbors.len() - 1).rev() {
                            neighbor_of_neighbors[i + 1] = neighbor_of_neighbors[i];
                        }
                        neighbor_of_neighbors[insert_pos] = local_node;
                    }
                }
            })
            .count()
    }

    pub fn improve_neighborhoods(&mut self) {
        for layer_id_from_top in 0..self.layer_count() {
            eprintln!("improving {layer_id_from_top}");
            let count = self.link_layer_to_better_neighbors(layer_id_from_top);
            eprintln!("{layer_id_from_top}: relinked {count}");
        }
    }

    pub fn improve_index(&mut self) {
        for layer_id in 0..self.layer_count() - 1 {
            let vecs = self.discover_vectors_to_promote(layer_id);
            //eprintln!("Vectors to promote: {vecs:?}");
            self.extend_layer(layer_id + 1, vecs)
        }

        // final step: maybe we need a new top layer
        let mut vecs = self.discover_vectors_to_promote(self.layer_count() - 1);
        if !vecs.is_empty() {
            vecs.sort();
            eprintln!("promote {vecs:?}");
            assert!(vecs.len() < 24);
            let new_layer = self.generate_layer(self.comparator().clone(), vecs, 24);
            eprintln!("done generating cool new layer");
            eprintln!("{:?}", new_layer.nodes);
            eprintln!("{:?}", new_layer.neighbors);
            self.layers.insert(0, new_layer);
        }
    }
}

fn cross_compare_vectors<C: Comparator<T> + 'static, T: Sync + 'static>(
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

fn copy_old_neighbhoods_into_layer<C: Comparator<T> + 'static, T: Sync + 'static>(
    old_nodes: &[VectorId],
    old_neighbors: &[NodeId],
    old_nodes_map: &[usize],
    layer: &mut Layer<C, T>,
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

fn generate_node_maps<C: Comparator<T> + 'static, T: Sync + 'static>(
    mut vecs: Vec<VectorId>,
    layer: &mut Layer<C, T>,
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
            (Some(old), Some(new)) => {
                if old <= new {
                    layer.nodes.push(*old_nodes_iter.next().unwrap());
                    old_nodes_map.push(layer.nodes.len() - 1);
                } else {
                    vecs.push(**new);
                    layer.nodes.push(*new_nodes_iter.next().unwrap());
                    new_nodes_map.push(layer.nodes.len() - 1);
                }
            }
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

#[derive(Serialize, Deserialize, Debug)]
pub struct LayerMeta {
    pub node_count: usize,
    pub neighborhood_size: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HNSWMeta {
    pub layer_count: usize,
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

fn calculate_partitions(total_size: usize, neighborhood_size: usize) -> Vec<usize> {
    let mut partitions: Vec<usize> = vec![];
    let mut size = total_size;
    let order = neighborhood_size * 2;
    let layer_count = (total_size as f32).log(order as f32).ceil() as usize;
    for _ in 0..layer_count {
        partitions.push(size);
        size /= order;
    }
    partitions.reverse();
    partitions
}

pub fn make_random_hnsw(count: usize, dimension: usize) -> Hnsw<BigComparator, BigVec> {
    let data: Vec<Vec<f32>> = (0..count)
        .into_par_iter()
        .map(move |i| {
            let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
            random_normed_vec(&mut prng, dimension)
        })
        .collect();
    let c = BigComparator { data };
    let vs: Vec<_> = (0..count).map(VectorId).collect();
    let m = 24;
    let m0 = 48;
    let hnsw: Hnsw<BigComparator, BigVec> = Hnsw::generate(c, vs, m, m0);
    hnsw
}

pub type BigVec = Vec<f32>;
#[derive(Clone, Debug, PartialEq)]
pub struct BigComparator {
    pub data: Vec<BigVec>,
}

impl Comparator<BigVec> for BigComparator {
    type Params = ();
    fn compare_vec(&self, v1: AbstractVector<BigVec>, v2: AbstractVector<BigVec>) -> f32 {
        let v1 = match v1 {
            AbstractVector::Stored(i) => &self.data[i.0],
            AbstractVector::Unstored(v) => v,
        };
        let v2 = match v2 {
            AbstractVector::Stored(i) => &self.data[i.0],
            AbstractVector::Unstored(v) => v,
        };
        let mut result = 0.0;
        for (&f1, &f2) in v1.iter().zip(v2.iter()) {
            result += f1 * f2
        }
        1.0 - result
    }

    fn serialize<P: AsRef<Path>>(&self, _path: P) -> Result<(), SerializationError> {
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(_path: P, _: ()) -> Result<BigComparator, SerializationError> {
        Ok(BigComparator { data: Vec::new() })
    }
}

fn random_normed_vec(prng: &mut StdRng, size: usize) -> Vec<f32> {
    let range = Uniform::from(0.0..1.0);
    let vec: Vec<f32> = prng.sample_iter(&range).take(size).collect();
    let norm = vec.iter().map(|f| f * f).sum::<f32>().sqrt();
    let res = vec.iter().map(|f| f / norm).collect();
    res
}

#[cfg(test)]
mod tests {

    use super::*;
    type SillyVec = [f32; 3];
    #[derive(Clone, Debug, PartialEq)]
    struct SillyComparator {
        data: Vec<SillyVec>,
    }

    impl Comparator<SillyVec> for SillyComparator {
        type Params = ();
        fn compare_vec(&self, v1: AbstractVector<SillyVec>, v2: AbstractVector<SillyVec>) -> f32 {
            let v1 = match v1 {
                AbstractVector::Stored(i) => &self.data[i.0],
                AbstractVector::Unstored(v) => v,
            };
            let v2 = match v2 {
                AbstractVector::Stored(i) => &self.data[i.0],
                AbstractVector::Unstored(v) => v,
            };
            let mut result = 0.0;
            for (&f1, &f2) in v1.iter().zip(v2.iter()) {
                result += f1 * f2
            }
            1.0 - result
        }

        fn serialize<P: AsRef<Path>>(&self, _path: P) -> Result<(), SerializationError> {
            Ok(())
        }

        fn deserialize<P: AsRef<Path>>(
            _path: P,
            _: (),
        ) -> Result<SillyComparator, SerializationError> {
            Ok(SillyComparator { data: Vec::new() })
        }
    }

    fn make_simple_hnsw() -> Hnsw<SillyComparator, SillyVec> {
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

        let hnsw: Hnsw<SillyComparator, SillyVec> = Hnsw::generate(c, vs, 3, 6);
        hnsw
    }

    fn make_broken_hnsw() -> Hnsw<SillyComparator, SillyVec> {
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

        let mut hnsw: Hnsw<SillyComparator, SillyVec> = Hnsw::generate(c, vs, 3, 6);
        let bottom = &mut hnsw.layers[1];
        // add a ninth disconnected vector
        bottom.nodes.push(VectorId(9));
        bottom.neighbors.extend(vec![NodeId(!0); 6]);
        hnsw
    }
    #[test]
    fn test_nearness_search() {
        let hnsw: Hnsw<SillyComparator, SillyVec> = make_simple_hnsw();
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
        let hnsw: Hnsw<SillyComparator, SillyVec> = make_simple_hnsw();
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
        let hnsw: Hnsw<SillyComparator, SillyVec> = make_simple_hnsw();
        let data = &hnsw.layers[0].comparator.data;
        for (i, datum) in data.iter().enumerate() {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 9, 1);
            eprintln!("results: {results:?}");
            assert_eq!(VectorId(i), results[0].0)
        }
    }

    fn do_test_recall(hnsw: &Hnsw<BigComparator, BigVec>, minimum_recall: f32) {
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
                let results = hnsw.search(v, 300, 1);
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
    }

    #[test]
    fn test_supers() {
        let size = 10000;
        let dimension = 10;
        let hnsw: Hnsw<BigComparator, BigVec> = make_random_hnsw(size, dimension);
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
        let dimension = 100;
        let mut hnsw: Hnsw<BigComparator, BigVec> = make_random_hnsw(size, dimension);
        do_test_recall(&hnsw, 0.0);
        eprintln!("time to improve neighborhoods");
        hnsw.improve_neighborhoods();
        do_test_recall(&hnsw, 0.0);
        eprintln!("time to promote nodes");
        hnsw.improve_index();
        do_test_recall(&hnsw, 1.0);
        panic!();
    }

    #[test]
    fn test_small_index_improvement() {
        let mut hnsw: Hnsw<SillyComparator, SillyVec> = make_simple_hnsw();
        eprintln!("One from bottom: {:?}", hnsw.layers[hnsw.layer_count() - 2]);
        hnsw.improve_index();
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
        let mut hnsw: Hnsw<SillyComparator, SillyVec> = make_broken_hnsw();
        hnsw.improve_index();
        let data = &hnsw.layers[hnsw.layer_count() - 1].comparator.data;
        for (i, datum) in data.iter().enumerate() {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 9, 1);
            assert_eq!(VectorId(i), results[0].0)
        }
    }
}
