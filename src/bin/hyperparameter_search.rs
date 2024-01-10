use parallel_hnsw::{random_normed_vec, AbstractVector, BigComparator, BigVec, Hnsw, VectorId};
use rand::{rngs::StdRng, SeedableRng as _};
use rayon::prelude::*;

pub fn make_random_hnsw(
    count: usize,
    dimension: usize,
    neighborhood: usize,
) -> Hnsw<BigComparator, BigVec> {
    let data: Vec<Vec<f32>> = (0..count)
        .into_par_iter()
        .map(move |i| {
            let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
            random_normed_vec(&mut prng, dimension)
        })
        .collect();
    let c = BigComparator { data };
    let vs: Vec<_> = (0..count).map(VectorId).collect();
    let m = neighborhood;
    let m0 = neighborhood * 2;
    let hnsw: Hnsw<BigComparator, BigVec> = Hnsw::generate(c, vs, m, m0);
    hnsw
}

fn do_test_recall(
    hnsw: &Hnsw<BigComparator, BigVec>,
    candidates: usize,
    probe_depth: usize,
    supers: usize,
) -> usize {
    let data = &hnsw.comparator().data;
    let total_relevant: usize = data
        .par_iter()
        .enumerate()
        .map(|(i, datum)| {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, candidates, probe_depth, supers);
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
    let probe_depth: Vec<usize> = (1..5).collect();
    let supers: Vec<usize> = (1..10).collect();
    let mut neighborhood_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(neighborhood.len()).unwrap(),
    );
    let mut candidates_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(candidates.len()).unwrap(),
    );
    let mut probe_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(probe_depth.len()).unwrap(),
    );
    let mut supers_optimizer = tpe::TpeOptimizer::new(
        tpe::histogram_estimator(),
        tpe::categorical_range(supers.len()).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::from_seed(Default::default());
    let mut best_values = (0.0, 0.0, 0.0, 0.0);
    let mut best_value = std::f64::INFINITY;
    for _ in 0..100 {
        let neighborhood = neighborhood_optimizer.ask(&mut rng).unwrap();
        let candidates = candidates_optimizer.ask(&mut rng).unwrap();
        let supers = supers_optimizer.ask(&mut rng).unwrap();
        let probe_depth = probe_optimizer.ask(&mut rng).unwrap();
        eprintln!("trying neigborhood: {}", neighborhood as usize);
        eprintln!("trying candidates: {}", candidates as usize);
        eprintln!("trying supers: {}", supers as usize);
        eprintln!("trying probe_depth: {}", probe_depth as usize);
        let hnsw = make_random_hnsw(input_size, vector_size, neighborhood as usize);
        let total_relevant = do_test_recall(
            &hnsw,
            candidates as usize,
            probe_depth as usize,
            supers as usize,
        );
        let recall = total_relevant as f64 / input_size as f64;
        eprintln!("Obtained recall of: {recall}");
        let one_minus_recall = 1.0 - recall;
        neighborhood_optimizer
            .tell(neighborhood, one_minus_recall)
            .unwrap();
        candidates_optimizer
            .tell(candidates, one_minus_recall)
            .unwrap();
        supers_optimizer.tell(supers, one_minus_recall).unwrap();
        if one_minus_recall < best_value {
            best_values = (neighborhood, candidates, probe_depth, supers)
        }
        best_value = best_value.min(one_minus_recall);
    }

    let (neighborhood, candidates, probe_depth, supers) = best_values;

    println!("neighborhood: {neighborhood}");
    println!("candidates: {candidates}");
    println!("supers: {supers}");
    println!("probe_depth: {probe_depth}");
    println!("best_value: {best_value}");
}
