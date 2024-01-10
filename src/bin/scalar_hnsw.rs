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

impl ScalarComparator {
    fn value_for(&self, vec: AbstractVector<f32>) -> f32 {
        match vec {
            AbstractVector::Stored(i) => self.data[i.0],
            AbstractVector::Unstored(v) => *v,
        }
    }
}

impl Comparator<f32> for ScalarComparator {
    type Params = ();

    fn compare_vec(
        &self,
        v1: parallel_hnsw::AbstractVector<f32>,
        v2: parallel_hnsw::AbstractVector<f32>,
    ) -> f32 {
        (self.value_for(v1) - self.value_for(v2)).abs()
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

fn do_test_recall(hnsw: &Hnsw<ScalarComparator, f32>) -> Vec<usize> {
    let data = &hnsw.comparator().data;
    let unrecallable_vecs: Vec<_> = data
        .par_iter()
        .enumerate()
        .filter_map(|(i, datum)| {
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 300, 1, 1);
            if VectorId(i) == results[0].0 {
                None
            } else {
                Some(i)
            }
        })
        .collect();

    unrecallable_vecs
}

pub fn main() {
    let mut hnsw = make_random_scalar_hnsw(1000000);
    let unrecallable_vecs = do_test_recall(&hnsw);
    assert_ne!(0, unrecallable_vecs.len());
    /*
    let unrecallable_vec = unrecallable_vecs[0];
    let unrecallable_val = hnsw.comparator().data[unrecallable_vec];
    eprintln!("unrecallable value: {unrecallable_val} ({unrecallable_vec})");
    eprintln!("layer count: {}", hnsw.layer_count());
    dbg!(hnsw.search_noisy(AbstractVector::Unstored(&unrecallable_val), 300));
    let bottom_layer = hnsw.get_layer(0).unwrap();
    let bottom_node = bottom_layer.get_node(VectorId(unrecallable_vec)).unwrap();
    let reverse_neighbors = bottom_layer.reverse_get_neighbors(bottom_node);
    eprintln!("should have been reachable from {:?}", reverse_neighbors);
    let mut reverse_neighbor_values: Vec<_> = reverse_neighbors
        .iter()
        .map(|n| bottom_layer.get_vector(*n))
        .map(|v| {
            (
                v,
                bottom_layer.comparator.compare_vec(
                    AbstractVector::Stored(v),
                    AbstractVector::Unstored(&unrecallable_val),
                ),
            )
        })
        .collect();
    reverse_neighbor_values.sort_by_key(|(_, d)| OrderedFloat(*d));
    eprintln!(
        "should have been reachable from {:?}",
        reverse_neighbor_values
    );
    */

    /*
    for i in 0..hnsw.layer_count() {
        eprintln!("checking layer {i}");
        hnsw.discover_vectors_to_promote(i);
    }
    */

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
