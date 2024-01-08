use std::time::Instant;

use parallel_hnsw::{make_random_hnsw, AbstractVector, BigComparator, BigVec, Hnsw, VectorId};
fn do_test_recall(hnsw: &Hnsw<BigComparator, BigVec>) -> f32 {
    let data = &hnsw.comparator().data;
    let total = data.len();
    let mut total_relevant = 0;
    for (i, datum) in data.iter().enumerate() {
        let v = AbstractVector::Unstored(datum);
        let results = hnsw.search(v, 300);
        if VectorId(i) == results[0].0 {
            total_relevant += 1;
        }
    }
    total_relevant as f32 / total as f32
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
            let recall = do_test_recall(&hnsw);
            let recall_elapsed = start.elapsed();

            println!(
                "{vector_size},{input_size},{recall},{},{}",
                hnsw_elapsed.as_millis(),
                recall_elapsed.as_millis()
            );
        }
    }
}
