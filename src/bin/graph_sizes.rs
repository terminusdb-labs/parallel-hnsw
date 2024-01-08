use std::time::Instant;

use parallel_hnsw::{make_random_hnsw, AbstractVector, BigComparator, BigVec, Hnsw, VectorId};
use rayon::prelude::*;
fn do_test_recall(hnsw: &Hnsw<BigComparator, BigVec>) -> usize {
    let data = &hnsw.comparator().data;
    let total = data.len();
    let total_relevant: usize = data
        .par_iter()
        .enumerate()
        .map(|(i, datum)| {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 300);
            if VectorId(i) == results[0].0 {
                1
            } else {
                0
            }
        })
        .sum();
    total_relevant
}

pub fn main() {
    let vector_sizes = vec![10, 100, 1000];
    let input_sizes: Vec<usize> = (1..=100).map(|i| i * 1000).collect();

    for vector_size in vector_sizes {
        for input_size in input_sizes.iter() {
            let start = Instant::now();
            let hnsw = make_random_hnsw(*input_size, vector_size);
            let hnsw_elapsed = start.elapsed();
            let start = Instant::now();
            let total_relevant = do_test_recall(&hnsw);
            let recall_elapsed = start.elapsed();

            println!(
                "{vector_size},{input_size},{total_relevant},{},{}",
                hnsw_elapsed.as_millis(),
                recall_elapsed.as_millis()
            );
        }
    }
}
