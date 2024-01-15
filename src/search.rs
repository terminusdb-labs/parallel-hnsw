use rayon::prelude::*;
use std::marker::PhantomData;

use crate::{
    priority_queue::PriorityQueue, AbstractVector, Comparator, Layer, NodeDistances, NodeId,
    OrderedFloat, VectorId,
};

#[derive(PartialEq, PartialOrd, Debug, Clone)]
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

impl<C: Comparator<T>, T> HnswSearcher<C, T> {
    pub fn entry_vector<L: AsRef<Layer<C, T>>>(layers: &[L]) -> VectorId {
        layers[0].as_ref().nodes[0]
    }

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

    pub fn generate_initial_partitions<L: AsRef<Layer<C, T>> + Sync>(
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
                    Self::compare_all(comparator, *vector_id, vs)
                } else {
                    //eprintln!("not empty layers");
                    Self::initial_vector_distances(*vector_id, number_of_supers_to_check, layers)
                };
                //eprintln!("ivd: {initial_vector_distances:?}");
                let initial_node_distances: Vec<_> = initial_vector_distances
                    .into_iter()
                    .map(|(inner_vector_id, distance)| {
                        (
                            NodeId(vs.binary_search(&inner_vector_id).unwrap()),
                            distance,
                        )
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
        v: VectorId,
        number_of_nodes: usize,
        layers: &[L],
    ) -> Vec<(VectorId, f32)> {
        Self::search_layers(AbstractVector::Stored(v), number_of_nodes, layers, 1)
            .into_iter()
            .filter(|(w, _)| v != *w)
            .collect::<Vec<_>>()
    }

    pub fn search_layers<L: AsRef<Layer<C, T>>>(
        v: AbstractVector<T>,
        number_of_candidates: usize,
        layers: &[L],
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        Self::search_layers_noisy(v, number_of_candidates, layers, probe_depth, false).0
    }

    pub fn search_layers_noisy<L: AsRef<Layer<C, T>>>(
        v: AbstractVector<T>,
        number_of_candidates: usize,
        layers: &[L],
        probe_depth: usize,
        noisy: bool,
    ) -> (Vec<(VectorId, f32)>, usize) {
        let upper_layer_candidate_count = 1;
        let entry_vector = Self::entry_vector(layers);
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
        let mut last_index_distance = usize::MAX;
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
            let (closest, index_distance) = layer.as_ref().closest_vectors(
                v.clone(),
                &candidates,
                candidate_count,
                probe_depth,
            );
            last_index_distance = index_distance;
            if noisy {
                eprintln!("closest: {closest:?}");
            }
            candidates.merge_pairs(&closest);
        }

        (candidates.iter().collect(), last_index_distance)
    }
}
