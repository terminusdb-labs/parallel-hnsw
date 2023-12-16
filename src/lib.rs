use std::{cell::UnsafeCell, collections::HashSet, marker::PhantomData};

use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp};
use rayon::prelude::*;

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct VectorId(pub usize);
#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct NodeId(pub usize);

pub enum AbstractVector<'a, T> {
    Stored(VectorId),
    Unstored(&'a T),
}

impl<'a, T> Clone for AbstractVector<'a, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Stored(arg0) => Self::Stored(*arg0),
            Self::Unstored(arg0) => Self::Unstored(arg0),
        }
    }
}

pub trait Comparator<T>: Sync + Clone {
    fn compare_vec(&self, v1: AbstractVector<T>, v2: AbstractVector<T>) -> f32;
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Layer<C: Comparator<T>, T> {
    comparator: C,
    neighborhood_size: usize,
    nodes: Vec<VectorId>,
    neighbors: Vec<NodeId>,
    _phantom: PhantomData<T>,
}

impl<C: Comparator<T>, T> Layer<C, T> {
    #[allow(unused)]
    fn get_node(&self, v: VectorId) -> Option<NodeId> {
        self.nodes.binary_search(&v).ok().map(NodeId)
    }

    fn get_vector(&self, n: NodeId) -> VectorId {
        self.nodes[n.0]
    }

    fn get_neighbors(&self, n: NodeId) -> &[NodeId] {
        &self.neighbors[(n.0 * self.neighborhood_size)..((n.0 + 1) * self.neighborhood_size)]
    }

    pub fn closest_nodes(
        &self,
        v: AbstractVector<T>,
        number_of_nodes: usize,
    ) -> Vec<(NodeId, f32)> {
        let mut result: Vec<(NodeId, f32)> = vec![(
            NodeId(0),
            self.comparator.compare_vec(
                v.clone(),
                AbstractVector::Stored(self.get_vector(NodeId(0))),
            ),
        )];
        let mut visit_queue = vec![(NodeId(0), f32::MAX)];
        let mut visited: HashSet<NodeId> = HashSet::new();
        while let Some((next, _)) = visit_queue.pop() {
            visited.insert(next);
            let worst = result.last().cloned();
            let neighbors = self.get_neighbors(next);
            let neighbor_distances: Vec<_> = neighbors
                .iter() // Remove empty cells and previously visited nodes
                .filter(|n| n.0 != !0 && !visited.contains(*n))
                .map(|n| {
                    (
                        *n,
                        self.comparator
                            .compare_vec(v.clone(), AbstractVector::Stored(self.get_vector(*n))),
                    )
                })
                .collect();
            visited.extend(neighbor_distances.iter().map(|(n, _)| n));
            visit_queue.extend(
                neighbor_distances
                    .iter()
                    .filter(|(_, d)| worst.is_none() || worst.as_ref().unwrap().1 > *d),
            );

            result.extend(neighbor_distances);
            result.sort_by_key(|(_, distance)| OrderedFloat(*distance));
            result.truncate(number_of_nodes);

            if result.len() == number_of_nodes && worst == result.last().cloned() {
                break;
            }
            visit_queue.sort_by_key(|(_, distance)| OrderedFloat(*distance));
        }

        result
    }

    pub fn closest_vectors(
        &self,
        v: AbstractVector<T>,
        number_of_vectors: usize,
    ) -> Vec<(VectorId, f32)> {
        self.closest_nodes(v, number_of_vectors)
            .iter()
            .map(|(node_id, distance)| (self.get_vector(*node_id), *distance))
            .collect()
    }

    pub fn closest_vector(&self, v: AbstractVector<T>) -> (VectorId, f32) {
        let (node_id, distance) = self.closest_nodes(v, 1)[0];
        (self.get_vector(node_id), distance)
    }
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Hnsw<C: Comparator<T>, T: Sync> {
    layers: Vec<Layer<C, T>>,
}

impl<C: Comparator<T>, T: Sync> Hnsw<C, T> {
    pub fn get_layer(&self, i: usize) -> Option<&Layer<C, T>> {
        if self.layer_count() > i {
            Some(&self.layers[self.layer_count() - i - 1])
        } else {
            None
        }
    }

    pub fn get_layer_above(&self, i: usize) -> Option<&Layer<C, T>> {
        self.get_layer(i + 1)
    }

    pub fn initial_vector_distances(
        &self,
        v: VectorId,
        level: usize,
        number_of_nodes: usize,
    ) -> Vec<(VectorId, f32)> {
        self.get_layer_above(level)
            .map(|previous_layer| {
                previous_layer.closest_vectors(AbstractVector::Stored(v), number_of_nodes)
            })
            .unwrap_or_default()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn search(
        &self,
        v: AbstractVector<T>,
        number_of_candidates: usize,
    ) -> Vec<(VectorId, f32)> {
        let upper_layer_candidate_count = 1;
        let mut candidates_queue = Vec::new();
        for i in 0..self.layer_count() {
            let candidate_count = if i == self.layer_count() - 1 {
                number_of_candidates
            } else {
                upper_layer_candidate_count
            };
            let layer = &self.layers[i];
            let closest = layer.closest_vectors(v.clone(), candidate_count);
            candidates_queue.extend(closest);
            candidates_queue.sort_by_key(|(_, d)| OrderedFloat(*d));
            candidates_queue.dedup();
            candidates_queue.truncate(number_of_candidates);
        }
        candidates_queue
    }

    pub fn generate_layer(
        &self,
        comparator: C,
        vs: Vec<VectorId>,
        level: usize,
        neighborhood_size: usize,
    ) -> Layer<C, T> {
        // Parameter for the number of neighbours to look at from above.
        let number_of_supers_to_check = 3;

        // 1. Calculate our node id, and find our neighborhood in the above layer
        let mut initial_partitions: Vec<_> = vs
            .par_iter()
            .enumerate()
            .map(|(node_id, vector_id)| {
                let initial_vector_distances =
                    self.initial_vector_distances(*vector_id, level, number_of_supers_to_check);
                let initial_node_distances: Vec<_> = initial_vector_distances
                    .into_iter()
                    .map(|(vector_id, distance)| {
                        (NodeId(vs.binary_search(&vector_id).unwrap()), distance)
                    })
                    .collect();
                (NodeId(node_id), *vector_id, initial_node_distances)
            })
            .collect();
        initial_partitions.par_sort_unstable_by_key(|(_node_id, _vector_id, distances)| {
            distances.first().map(|(_, d)| OrderedFloat(*d))
        });

        // 2. Partition the layer in terms of the closeness to the
        // best node in the layer above
        let partition_groups = initial_partitions
            .into_iter()
            .into_group_map_by(|(_, _, distances)| distances.first().map(|(id, _)| *id));

        // 3. Calculate our neighbourhoods by comparing distances in our partition
        let borrowed_comparator = &comparator;
        let mut all_distances: Vec<UnsafeCell<Vec<(NodeId, f32)>>> = partition_groups
            .par_iter()
            .flat_map(|(_sup, partition)| {
                let max = partition.len();
                partition
                    .par_iter()
                    .map(|(node_id, vector_id, distances)| {
                        let mut distances = distances.clone();
                        let super_nodes: Vec<_> =
                            distances.iter().map(|(node, _)| node).cloned().collect();

                        // some random, some for neighborhood
                        // TODO - also some random extra nodes on the same layer
                        let number_of_nodes_to_check =
                            std::cmp::min(neighborhood_size * 10, max.saturating_sub(1));
                        let choice_count = std::cmp::min(
                            number_of_nodes_to_check.saturating_sub(distances.len()),
                            max.saturating_sub(1),
                        );
                        let mut prng = StdRng::seed_from_u64(
                            level as u64 + vector_id.0 as u64 + vs.len() as u64,
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
                        distances.truncate(neighborhood_size);
                        UnsafeCell::new(distances)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // 4. Make neighborhoods bidirectional
        for i in 0..all_distances.len() {
            for (n, d) in unsafe { &*(all_distances[i].get()) } {
                // Was an assertion, but we're looking in other partitions now
                // debug_assert!(n.0 != i);
                if n.0 != i {
                    let other = all_distances[n.0].get_mut();
                    if !other.iter().any(|pair| pair.0 == NodeId(i)) {
                        other.push((NodeId(i), *d));
                    }
                }
            }
        }

        // This neighbors array, despite seemingly immutable, is going
        // to be mutated unsafely! However, each segment is logically
        // independent and therefore safe.
        let neighbors = vec![NodeId(!0); vs.len() * neighborhood_size];

        // 5. In parallel, write our own best neighbors, in order of
        // distance, into our neighborhood array truncating to
        // neighborhood size
        all_distances
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, distances)| {
                let distances = distances.get_mut();
                distances.sort_by_key(|d| (OrderedFloat(d.1), d.0));
                distances.truncate(neighborhood_size);
                // We know we have a unique index here that is not
                // going to be contended. Therefore we just use
                // unsafe.
                let unsafe_neighbors: *mut NodeId = neighbors.as_ptr() as *mut NodeId;
                (0..distances.len()).for_each(|j| unsafe {
                    let offset = unsafe_neighbors.add(i * neighborhood_size + j);
                    *offset = distances[j].0;
                });
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
        eprintln!("neighborhood_size: {neighborhood_size}");
        eprintln!("total_size: {total_size}");
        let layer_count = (total_size as f32).log(neighborhood_size as f32).ceil() as usize;
        eprintln!("layer count: {layer_count}");
        let layers = Vec::with_capacity(layer_count);
        let mut hnsw: Hnsw<C, T> = Hnsw { layers };
        for i in 0..layer_count {
            let level = layer_count - i - 1;
            eprintln!("Generating level {level}");
            let layer_size = neighborhood_size.pow(i as u32 + 1);
            let slice_length = std::cmp::min(layer_size, vs.len());
            let slice = &vs[0..slice_length];
            let neighbors = if level == 0 {
                zero_layer_neighborhood_size
            } else {
                neighborhood_size
            };
            let layer = hnsw.generate_layer(c.clone(), slice.to_vec(), level, neighbors);
            hnsw.layers.push(layer)
        }

        hnsw
    }
}

fn choose_n(
    n: usize,
    partition_maxes: Vec<usize>,
    exclude: usize,
    prng: &mut StdRng,
) -> Vec<(usize, usize)> {
    // todo: probably should give higher chance to select our own partition
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

#[cfg(test)]
mod tests {

    use rand_distr::Uniform;

    use super::*;
    type SillyVec = [f32; 3];
    #[derive(Clone, Debug, PartialEq)]
    struct SillyComparator {
        data: Vec<SillyVec>,
    }

    impl Comparator<SillyVec> for SillyComparator {
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
    }

    fn make_simple_hnsw() -> Hnsw<SillyComparator, SillyVec> {
        let sqrt2_recip = std::f32::consts::FRAC_1_SQRT_2;
        let data: Vec<SillyVec> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [sqrt2_recip, sqrt2_recip, 0.0],
            [0.5773, 0.5773, 0.5773],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, sqrt2_recip, sqrt2_recip],
        ];
        let c = SillyComparator { data: data.clone() };
        let vs: Vec<_> = (0..9).map(VectorId).collect();

        let hnsw: Hnsw<SillyComparator, SillyVec> = Hnsw::generate(c, vs, 3, 6);
        hnsw
    }

    type BigVec = Vec<f32>;
    #[derive(Clone, Debug, PartialEq)]
    struct BigComparator {
        data: Vec<BigVec>,
    }

    impl Comparator<BigVec> for BigComparator {
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
    }

    fn random_normed_vec(prng: &mut StdRng, size: usize) -> Vec<f32> {
        let range = Uniform::from(0.0..1.0);
        let vec: Vec<f32> = prng.sample_iter(&range).take(size).collect();
        let norm = vec.iter().map(|f| f * f).sum::<f32>().sqrt();
        let res = vec.iter().map(|f| f / norm).collect();
        res
    }

    fn make_random_hnsw(count: usize, dimension: usize) -> Hnsw<BigComparator, BigVec> {
        let data: Vec<Vec<f32>> = (0..count)
            .into_par_iter()
            .map(move |i| {
                let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
                random_normed_vec(&mut prng, dimension)
            })
            .collect();
        let c = BigComparator { data };
        let vs: Vec<_> = (0..count).map(VectorId).collect();

        let hnsw: Hnsw<BigComparator, BigVec> = Hnsw::generate(c, vs, 24, 48);
        hnsw
    }

    #[test]
    fn test_nearness_search() {
        let hnsw: Hnsw<SillyComparator, SillyVec> = make_simple_hnsw();
        let sqrt2_recip = std::f32::consts::FRAC_1_SQRT_2;
        let slice = &[0.0, sqrt2_recip, sqrt2_recip];
        let search_vector = AbstractVector::Unstored(slice);
        let results = hnsw.search(search_vector, 9);
        assert_eq!(results, vec![])
    }

    #[test]
    fn test_generation() {
        let hnsw: Hnsw<SillyComparator, SillyVec> = make_simple_hnsw();
        assert_eq!(
            hnsw.get_layer(1).map(|layer| &layer.nodes),
            Some(vec![VectorId(0), VectorId(1), VectorId(2)].as_ref())
        );
        assert_eq!(
            hnsw.get_layer(1).map(|layer| &layer.neighbors),
            Some(
                vec![
                    NodeId(1),
                    NodeId(2),
                    NodeId(18446744073709551615),
                    NodeId(0),
                    NodeId(2),
                    NodeId(18446744073709551615),
                    NodeId(0),
                    NodeId(1),
                    NodeId(18446744073709551615)
                ]
                .as_ref()
            )
        );
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
                    // Node 0
                    NodeId(3),
                    NodeId(4),
                    NodeId(1),
                    NodeId(2),
                    NodeId(6),
                    NodeId(7),
                    // Node 1
                    NodeId(3),
                    NodeId(8),
                    NodeId(4),
                    NodeId(0),
                    NodeId(2),
                    NodeId(5),
                    // Node 2
                    NodeId(8),
                    NodeId(4),
                    NodeId(0),
                    NodeId(1),
                    NodeId(3),
                    NodeId(5),
                    // Node 3
                    NodeId(4),
                    NodeId(0),
                    NodeId(1),
                    NodeId(8),
                    NodeId(2),
                    NodeId(7),
                    // Node 4
                    NodeId(3),
                    NodeId(8),
                    NodeId(0),
                    NodeId(1),
                    NodeId(2),
                    NodeId(5),
                    // Node 5
                    NodeId(1),
                    NodeId(2),
                    NodeId(6),
                    NodeId(7),
                    NodeId(8),
                    NodeId(4),
                    // Node 6
                    NodeId(0),
                    NodeId(2),
                    NodeId(5),
                    NodeId(7),
                    NodeId(4),
                    NodeId(3),
                    // Node 7
                    NodeId(0),
                    NodeId(1),
                    NodeId(3),
                    NodeId(5),
                    NodeId(6),
                    NodeId(4),
                    // Node 8
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
            let results = hnsw.search(v, 9);
            assert_eq!(VectorId(i), results[0].0)
        }
    }

    #[test]
    fn test_recall() {
        let size = 10000;
        let dimension = 10;
        let hnsw: Hnsw<BigComparator, BigVec> = make_random_hnsw(size, dimension);
        let data = &hnsw.layers[0].comparator.data;
        let total = data.len();
        let mut total_relevant = 0;
        for (i, datum) in data.iter().enumerate() {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 50);
            if VectorId(i) == results[0].0 {
                total_relevant += 1;
            }
        }
        eprintln!("total relevant: {total_relevant}");
        eprintln!("from total: {total}");
        let recall = total_relevant as f32 / total as f32;
        eprintln!("with recall: {recall}");
        assert!(recall > 0.9)
    }
}
