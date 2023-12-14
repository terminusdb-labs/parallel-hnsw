#[macro_use]
extern crate timeit;

use std::{cell::UnsafeCell, collections::HashSet, marker::PhantomData};

use rand::{thread_rng, Rng};
use rayon::prelude::*;

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy)]
pub struct VectorId(usize);
#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy)]
pub struct NodeId(usize);

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

    fn get_neighbours(&self, n: NodeId) -> &[NodeId] {
        &self.neighbors[(n.0 * NEIGHBORHOOD_SIZE)..((n.0 + 1) * NEIGHBORHOOD_SIZE)]
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

        let mut neighbors = vec![NodeId(0); vs.len() * NEIGHBORHOOD_SIZE];
        all_distances
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, distances)| {
                let distances = distances.get_mut();
                distances.sort_by_key(|d| OrderedFloat(d.1));
                distances.dedup();
                distances.truncate(NEIGHBORHOOD_SIZE);
                for j in 0..distances.len() {
                    neighbors[i * NEIGHBORHOOD_SIZE + j] = NodeId(distances[j].0);
                }
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
    use std::sync::{
        atomic::{self, AtomicUsize},
        Arc,
    };

    use super::*;
    type SillyVec = [f32; 8];
    #[derive(Clone)]
    struct SillyComparator {
        data: Vec<SillyVec>,
    }

    impl Comparator<SillyVec> for SillyComparator {
        fn compare_stored(&self, v1: VectorId, v2: VectorId) -> f32 {
            let v1 = &self.data[v1.0];
            let v2 = &self.data[v2.0];
            self.compare_unstored(v1, v2)
        }

        fn compare_half_stored(&self, v1: VectorId, v2: &SillyVec) -> f32 {
            let v1 = &self.data[v1.0];
            self.compare_unstored(v1, v2)
        }

        fn compare_unstored(&self, v1: &SillyVec, v2: &SillyVec) -> f32 {
            let mut result = 0.0;
            for (&f1, &f2) in v1.iter().zip(v2.iter()) {
                result += f1 * f2;
            }
            result
        }
    }

    fn generate_random_vector() -> [f32; 8] {
        let mut rng = thread_rng();
        rng.gen()
    }

    fn create_test_data() -> SillyComparator {
        let mut vec = Vec::new();
        for _ in 0..10000 {
            vec.push(generate_random_vector());
        }
        SillyComparator { data: vec }
    }

    #[test]
    fn it_works() {
        let comparator = create_test_data();
        let vs: Vec<VectorId> = (0..10000).map(VectorId).collect();
        timeit!({
            let result: Layer<10, _, _> = Layer::generate(comparator.clone(), vs.clone());
            drop(result);
        });
        panic!();
    }
}
