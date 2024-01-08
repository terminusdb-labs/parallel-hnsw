use parallel_hnsw::{
    make_random_hnsw, AbstractVector, Comparator, Hnsw, NodeId, OrderedFloat, VectorId,
};
use rand::{rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::prelude::*;

#[derive(Clone)]
struct ScalarComparator {
    data: Vec<f32>,
}

impl Comparator<f32> for ScalarComparator {
    type Params = ();

    fn compare_vec(
        &self,
        v1: parallel_hnsw::AbstractVector<f32>,
        v2: parallel_hnsw::AbstractVector<f32>,
    ) -> f32 {
        match (v1, v2) {
            (
                parallel_hnsw::AbstractVector::Stored(i1),
                parallel_hnsw::AbstractVector::Stored(i2),
            ) => (self.data[i1.0] - self.data[i2.0]).abs(),
            (
                parallel_hnsw::AbstractVector::Stored(i),
                parallel_hnsw::AbstractVector::Unstored(v),
            )
            | (
                parallel_hnsw::AbstractVector::Unstored(v),
                parallel_hnsw::AbstractVector::Stored(i),
            ) => (v - self.data[i.0]).abs(),
            (
                parallel_hnsw::AbstractVector::Unstored(v1),
                parallel_hnsw::AbstractVector::Unstored(v2),
            ) => (v1 - v2).abs(),
        }
    }

    fn serialize<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), parallel_hnsw::SerializationError> {
        todo!()
    }

    fn deserialize<P: AsRef<std::path::Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, parallel_hnsw::SerializationError> {
        todo!()
    }
}

fn make_random_scalar_hnsw(count: usize) -> Hnsw<ScalarComparator, f32> {
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::from(-10.0..10.0);
    let mut data: Vec<f32> = rng.sample_iter(&range).take(count).collect();
    data.sort_by_key(|x| OrderedFloat(*x));
    data.dedup();
    let mut rng = StdRng::seed_from_u64(42);
    data.shuffle(&mut rng);
    let count = data.len();

    let c = ScalarComparator { data };
    let vs: Vec<_> = (0..count).map(VectorId).collect();
    let m = 24;
    let m0 = 48;
    let hnsw: Hnsw<ScalarComparator, f32> = Hnsw::generate(c, vs, m, m0);
    hnsw
}

fn do_test_recall(hnsw: &Hnsw<ScalarComparator, f32>) -> usize {
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
    let mut hnsw = make_random_scalar_hnsw(100000);
    let total_relevant = do_test_recall(&hnsw);
    eprintln!("{total_relevant} of {}", hnsw.comparator().data.len());
    hnsw.improve_index();
    let total_relevant = do_test_recall(&hnsw);
    eprintln!("{total_relevant} of {}", hnsw.comparator().data.len());
    /*
    eprintln!("data: {:?}", hnsw.comparator().data);
    for i in 0..hnsw.layer_count() {
        let layer_i = hnsw.get_layer_from_top(i).unwrap();
        eprintln!("layer {i}");
        eprintln!(
            " nodes: {:?}",
            layer_i.nodes.iter().map(|n| n.0).collect::<Vec<_>>()
        );
        for node in 0..layer_i.nodes.len() {
            let node = NodeId(node);
            eprintln!(
                " neighbors for {}: {:?}",
                node.0,
                layer_i
                    .get_neighbors(node)
                    .iter()
                    .map(|n| n.0)
                    .filter(|n| *n != !0)
                    .collect::<Vec<_>>()
            );
        }
    }
    */
}
