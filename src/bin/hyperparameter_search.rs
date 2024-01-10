use std::time::Instant;

use parallel_hnsw::{make_random_hnsw, AbstractVector, BigComparator, BigVec, Hnsw, VectorId};
use rand::SeedableRng as _;
use rayon::prelude::*;

fn do_test_recall(hnsw: &Hnsw<BigComparator, BigVec>) -> usize {
    let data = &hnsw.comparator().data;
    let total = data.len();
    let total_relevant: usize = data
        .par_iter()
        .enumerate()
        .map(|(i, datum)| {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 300, 1, 1);
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
    let vector_size = 1000;
    let input_size = 10000;
    let neighborhood: Vec<usize> = (10..30).map(|n| n * 2).collect();
    let candidates: Vec<usize> = (1..10).map(|n| n * 100).collect();
    let supers: Vec<usize> = (1..10).collect();
    let mut neighborhood_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(neighborhood.len()).unwrap(),
    );
    let mut candidates_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(candidates.len()).unwrap(),
    );
    let mut supers_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(supers.len()).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::from_seed(Default::default());
    let mut best_value = std::f64::INFINITY;
    for _ in 0..100 {
        let neighborhood = neighborhood_optimizer.ask(&mut rng).unwrap();
        let candidates = candidates_optimizer.ask(&mut rng).unwrap();
        let supers = supers_optimizer.ask(&mut rng).unwrap();
        let hnsw = make_random_hnsw(input_size, vector_size);
        let total_relevant = do_test_recall(&hnsw);
        let recall = total_relevant as f64 / input_size as f64;
        neighborhood_optimizer.tell(neighborhood, recall).unwrap();
        candidates_optimizer.tell(candidates, recall).unwrap();
        supers_optimizer.tell(supers, recall).unwrap();
        best_value = best_value.min(recall);
    }

    let neighborhood = neighborhood_optimizer.ask(&mut rng).unwrap();
    let candidates = candidates_optimizer.ask(&mut rng).unwrap();
    let supers = supers_optimizer.ask(&mut rng).unwrap();

    println!("neighborhood: {neighborhood}");
    println!("candidates: {candidates}");
    println!("supers: {supers}");
    println!("best_value: {best_value}");
}
