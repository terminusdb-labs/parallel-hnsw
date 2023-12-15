use std::{cell::UnsafeCell, collections::HashSet, marker::PhantomData};

use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
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

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct Layer<const NEIGHBORHOOD_SIZE: usize, C: Comparator<T>, T> {
    comparator: C,
    nodes: Vec<VectorId>,
    neighbors: Vec<NodeId>,
    _phantom: PhantomData<T>,
}

impl<const NEIGHBORHOOD_SIZE: usize, C: Comparator<T>, T> Layer<NEIGHBORHOOD_SIZE, C, T> {
    #[allow(unused)]
    fn get_node(&self, v: VectorId) -> Option<NodeId> {
        self.nodes.binary_search(&v).ok().map(NodeId)
    }

    fn get_vector(&self, n: NodeId) -> VectorId {
        self.nodes[n.0]
    }

    fn get_neighbors(&self, n: NodeId) -> &[NodeId] {
        &self.neighbors[(n.0 * NEIGHBORHOOD_SIZE)..((n.0 + 1) * NEIGHBORHOOD_SIZE)]
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
pub struct Hnsw<const NEIGHBORHOOD_SIZE: usize, C: Comparator<T>, T: Sync> {
    layers: Vec<Layer<NEIGHBORHOOD_SIZE, C, T>>,
}

impl<const NEIGHBORHOOD_SIZE: usize, C: Comparator<T>, T: Sync> Hnsw<NEIGHBORHOOD_SIZE, C, T> {
    pub fn get_layer(&self, i: usize) -> Option<&Layer<NEIGHBORHOOD_SIZE, C, T>> {
        if self.layer_count() > i {
            Some(&self.layers[self.layer_count() - i - 1])
        } else {
            None
        }
    }

    pub fn get_layer_above(&self, i: usize) -> Option<&Layer<NEIGHBORHOOD_SIZE, C, T>> {
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
    ) -> Layer<NEIGHBORHOOD_SIZE, C, T> {
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
            .into_par_iter()
            .flat_map(|(_sup, partition)| {
                let max = partition.len();
                partition
                    .par_iter()
                    .map(|(node_id, vector_id, distances)| {
                        let mut distances = distances.clone();
                        // some random, some for neighborhood
                        // TODO - also some random extra nodes on the same layer
                        let number_of_nodes_to_check =
                            std::cmp::min(NEIGHBORHOOD_SIZE * 10, max - 1);
                        let choice_count =
                            std::cmp::min(number_of_nodes_to_check - distances.len(), max - 1);
                        let prng = StdRng::seed_from_u64(
                            level as u64 + vector_id.0 as u64 + vs.len() as u64,
                        );

                        let partition_choices = choose_n(choice_count, max, node_id.0, prng);

                        for i in 0..partition_choices.len() {
                            let choice = &partition[partition_choices[i]];
                            let distance = borrowed_comparator.compare_vec(
                                AbstractVector::Stored(*vector_id),
                                AbstractVector::Stored(choice.1),
                            );
                            distances.push((choice.0, distance));
                        }
                        distances.sort_by_key(|d| (OrderedFloat(d.1), d.0));
                        distances.truncate(NEIGHBORHOOD_SIZE);
                        UnsafeCell::new(distances)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // 4. Make neighborhoods bidirectional
        for i in 0..all_distances.len() {
            for (n, d) in unsafe { &*(all_distances[i].get()) } {
                debug_assert!(n.0 != i);
                let other = all_distances[n.0].get_mut();
                if !other.iter().any(|pair| pair.0 == NodeId(i)) {
                    other.push((NodeId(i), *d));
                }
            }
        }

        // This neighbors array, despite seemingly immutable, is going
        // to be mutated unsafely! However, each segment is logically
        // independent and therefore safe.
        let neighbors = vec![NodeId(!0); vs.len() * NEIGHBORHOOD_SIZE];

        // 5. In parallel, write our own best neighbors, in order of
        // distance, into our neighborhood array truncating to
        // neighborhood size
        all_distances
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, distances)| {
                let distances = distances.get_mut();
                distances.sort_by_key(|d| (OrderedFloat(d.1), d.0));
                distances.truncate(NEIGHBORHOOD_SIZE);
                // We know we have a unique index here that is not
                // going to be contended. Therefore we just use
                // unsafe.
                let unsafe_neighbors: *mut NodeId = neighbors.as_ptr() as *mut NodeId;
                (0..distances.len()).for_each(|j| unsafe {
                    let offset = unsafe_neighbors.add(i * NEIGHBORHOOD_SIZE + j);
                    *offset = distances[j].0;
                });
            });

        Layer {
            comparator,
            nodes: vs,
            neighbors,
            _phantom: PhantomData,
        }
    }

    pub fn generate(c: C, vs: Vec<VectorId>) -> Self {
        let total_size = vs.len();
        let layer_count = (total_size as f32).log(NEIGHBORHOOD_SIZE as f32).ceil() as usize;
        let layers = Vec::with_capacity(layer_count);
        let mut hnsw: Hnsw<NEIGHBORHOOD_SIZE, C, T> = Hnsw { layers };
        for i in 0..layer_count {
            let level = layer_count - i - 1;
            let layer_size = NEIGHBORHOOD_SIZE.pow(i as u32 + 1);
            let slice_length = std::cmp::min(layer_size, vs.len());
            let slice = &vs[0..slice_length];
            let layer = hnsw.generate_layer(c.clone(), slice.to_vec(), level);
            hnsw.layers.push(layer)
        }

        hnsw
    }
}

fn choose_n(n: usize, max: usize, exclude: usize, mut prng: StdRng) -> Vec<usize> {
    let mut count = 0;
    let mut set = HashSet::with_capacity(n);
    while count != n {
        let selection = prng.gen_range(0..max);
        if selection != exclude && set.insert(selection) {
            count += 1;
        }
    }
    set.into_iter().collect()
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

    const SIMPLE_HNSW_SIZE: usize = 3;
    fn make_simple_hnsw() -> Hnsw<SIMPLE_HNSW_SIZE, SillyComparator, SillyVec> {
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

        let hnsw: Hnsw<SIMPLE_HNSW_SIZE, SillyComparator, SillyVec> = Hnsw::generate(c, vs);
        hnsw
    }
    #[test]
    fn test_nearness_search() {
        let hnsw: Hnsw<SIMPLE_HNSW_SIZE, SillyComparator, SillyVec> = make_simple_hnsw();
        let sqrt2_recip = std::f32::consts::FRAC_1_SQRT_2;
        let slice = &[0.0, sqrt2_recip, sqrt2_recip];
        let search_vector = AbstractVector::Unstored(slice);
        let results = hnsw.search(search_vector, 9);
        assert_eq!(results, vec![])
    }

    #[test]
    fn test_generation() {
        let hnsw: Hnsw<SIMPLE_HNSW_SIZE, SillyComparator, SillyVec> = make_simple_hnsw();
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
                    // Node 1
                    NodeId(3),
                    NodeId(8),
                    NodeId(4),
                    // Node 2
                    NodeId(8),
                    NodeId(4),
                    NodeId(0),
                    // Node 3
                    NodeId(4),
                    NodeId(0),
                    NodeId(1),
                    // Node 4
                    NodeId(3),
                    NodeId(8),
                    NodeId(0),
                    // Node 5
                    NodeId(1),
                    NodeId(2),
                    NodeId(6),
                    // Node 6
                    NodeId(0),
                    NodeId(2),
                    NodeId(5),
                    // Node 7
                    NodeId(0),
                    NodeId(1),
                    NodeId(3),
                    // Node 8
                    NodeId(4),
                    NodeId(1),
                    NodeId(2)
                ]
                .as_ref()
            )
        );
    }

    #[test]
    fn test_search() {
        let hnsw: Hnsw<SIMPLE_HNSW_SIZE, SillyComparator, SillyVec> = make_simple_hnsw();
        let data = &hnsw.layers[0].comparator.data;
        for (i, datum) in data.iter().enumerate() {
            eprintln!("Searching for VectorId({i})");
            let v = AbstractVector::Unstored(datum);
            let results = dbg!(hnsw.search(v, 9));
            assert_eq!(VectorId(i), results[0].0)
        }
    }
}
