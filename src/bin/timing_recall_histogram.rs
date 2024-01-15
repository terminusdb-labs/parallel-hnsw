use rayon::prelude::*;
use std::time::SystemTime;

use parallel_hnsw::{
    bigvec::{make_random_hnsw, BigComparator, BigVec},
    AbstractVector, Hnsw, VectorId,
};
fn do_test_recall(hnsw: &Hnsw<BigComparator, BigVec>) -> f32 {
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

    recall
}

pub fn main() {
    println!("\"dimensions\",\"count\",\"construction_time\",\"improvement_time\",\"improvement_iterations\",\"initial_recall\",\"final_recall\"");
    //let counts = [10_000, 100_000, 1_000_000];
    //let dimensions = [128, 768, 1024, 1536];
    let counts = [10_000_000];
    let dimensions = [1536];
    for dimension in dimensions {
        for count in counts.iter() {
            let start_time = SystemTime::now();
            let mut hnsw = make_random_hnsw(*count, dimension);
            let hnsw_construction_time = start_time.elapsed().unwrap().as_millis();
            let initial_recall = do_test_recall(&hnsw);
            let mut last_recall = initial_recall;
            let mut improvement = f32::MAX;
            let mut total_improvement_time = 0;
            let mut iteration = 0;
            while improvement > 0.001 {
                iteration += 1;
                let start_time = SystemTime::now();
                hnsw.improve_index();
                let hnsw_improvement_time = start_time.elapsed().unwrap();
                total_improvement_time += hnsw_improvement_time.as_millis();
                let new_recall = do_test_recall(&hnsw);
                improvement = new_recall - last_recall;
                last_recall = new_recall;
            }

            println!("{dimension},{count},{hnsw_construction_time},{total_improvement_time},{iteration},{initial_recall},{last_recall}");
        }
    }
}
