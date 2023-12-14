#[macro_use]
extern crate timeit;

use std::{cell::UnsafeCell, collections::HashSet, marker::PhantomData, sync::Mutex};

use rand::{thread_rng, Rng};
use rayon::prelude::*;

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct VectorId(pub usize);
#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct NodeId(pub usize);

pub trait Comparator<T>: Sync {
    fn compare_stored(&self, v1: VectorId, v2: VectorId) -> f32;
    fn compare_half_stored(&self, v1: VectorId, v2: &T) -> f32;
    fn compare_unstored(&self, v1: &T, v2: &T) -> f32;
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct Layer<const NEIGHBORHOOD_SIZE: usize, C: Comparator<T>, T> {
    comparator: C,
    nodes: Vec<VectorId>,
    neighbors: Vec<NodeId>,
    _phantom: PhantomData<T>,
}

impl<const NEIGHBORHOOD_SIZE: usize, C: Comparator<T>, T> Layer<NEIGHBORHOOD_SIZE, C, T> {
    fn get_node(&self, v: VectorId) -> NodeId {
        NodeId(self.nodes.binary_search(&v).unwrap())
    }

    fn get_vector(&self, n: NodeId) -> VectorId {
        self.nodes[n.0]
    }

    fn get_neighbors(&self, n: NodeId) -> &[NodeId] {
        &self.neighbors[(n.0 * NEIGHBORHOOD_SIZE)..((n.0 + 1) * NEIGHBORHOOD_SIZE)]
    }

    pub fn closest_nodes(&self, v: VectorId, number_of_nodes: usize) -> Vec<(NodeId, f32)> {
        let mut result: Vec<(NodeId, f32)> = Vec::new();
        let mut visit_queue = vec![NodeId(0)];
        let mut visited: HashSet<NodeId> = HashSet::new();
        while let Some(next) = visit_queue.pop() {
            visited.insert(next);
            let worst = result.last().cloned();
            let neighbors = self.get_neighbors(next);
            let neighbor_distances: Vec<_> = neighbors
                .iter()
                .enumerate()
                .filter(|(_ix, n)| !visited.contains(*n))
                .map(|(ix, n)| {
                    (
                        NodeId(ix),
                        self.comparator.compare_stored(v, self.get_vector(*n)),
                    )
                })
                .collect();
            visit_queue.extend(
                neighbor_distances
                    .iter()
                    .filter(|(_, d)| worst.is_none() || worst.as_ref().unwrap().1 > *d)
                    .map(|(node, _)| *node),
            );

            result.extend(neighbor_distances);
            result.sort_by_key(|(_, distance)| OrderedFloat(*distance));
            result.truncate(number_of_nodes);
            let new_worst = result.last().cloned();
            if worst == new_worst {
                break;
            }
        }

        result
    }

    pub fn closest_vector(&self, v: VectorId) -> (VectorId, f32) {
        let (node_id, distance) = self.closest_nodes(v, 1)[0];
        (self.get_vector(node_id), distance)
    }

    pub fn generate(comparator: C, vs: Vec<VectorId>) -> Self {
        let max = vs.len();

        let mut all_distances: Vec<_> = vs
            .par_iter()
            .map(|id| {
                let choices = choose_n(NEIGHBORHOOD_SIZE * 10, max, id.0);
                let mut distances = vec![(0, 0.0); NEIGHBORHOOD_SIZE * 10];
                for i in 0..choices.len() {
                    let choice = choices[i];
                    let distance = comparator.compare_stored(*id, VectorId(choice));
                    distances[i] = (choice, distance);
                }
                distances.sort_by_key(|d| OrderedFloat(d.1));
                distances.truncate(NEIGHBORHOOD_SIZE);

                UnsafeCell::new(distances)
            })
            .collect();

        for i in 0..all_distances.len() {
            for (n, d) in unsafe { &*(all_distances[i].get()) } {
                debug_assert!(*n != i);
                let other = all_distances[*n].get_mut();
                other.push((i, *d));
            }
        }

        // this neighbors, despite seemingly immutable, is going to be mutated unsafely!
        let neighbors = vec![NodeId(0); vs.len() * NEIGHBORHOOD_SIZE];
        all_distances
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, distances)| {
                let distances = distances.get_mut();
                distances.sort_by_key(|d| OrderedFloat(d.1));
                distances.dedup();
                distances.truncate(NEIGHBORHOOD_SIZE);
                // We know we have a unique index here that is not
                // going to be contended. Therefore we just use
                // unsafe.
                let unsafe_neighbors: *mut NodeId = neighbors.as_ptr() as *mut NodeId;
                (0..distances.len()).for_each(|j| unsafe {
                    let offset = unsafe_neighbors.add(i * NEIGHBORHOOD_SIZE + j);
                    *offset = NodeId(distances[j].0);
                });
            });

        Self {
            comparator,
            nodes: vs,
            neighbors,
            _phantom: PhantomData,
        }
    }
}

fn choose_n(n: usize, max: usize, exclude: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut count = 0;
    let mut set = HashSet::with_capacity(n);
    while count != n {
        let selection = rng.gen_range(0..max);
        if selection != exclude && set.insert(selection) {
            count += 1;
        }
    }

    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
}
