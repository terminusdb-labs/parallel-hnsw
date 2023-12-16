#![feature(test)]
extern crate test;

use parallel_hnsw::{AbstractVector, Comparator, Hnsw, Layer, VectorId};
use rand::{thread_rng, Rng};
use test::Bencher;
type SillyVec = [f32; 100];
#[derive(Clone)]
struct SillyComparator {
    data: Vec<SillyVec>,
}

impl Comparator<SillyVec> for SillyComparator {
    fn compare_vec(&self, v1: AbstractVector<SillyVec>, v2: AbstractVector<SillyVec>) -> f32 {
        let v1 = match v1 {
            AbstractVector::Stored(i) => &self.data[i.0],
            AbstractVector::Unstored(v) => v,
        };
        let v2 = match v2 {
            AbstractVector::Stored(i) => &self.data[i.0],
            AbstractVector::Unstored(v) => v,
        };
        let mut result = 0.0;
        for (&f1, &f2) in v1.iter().zip(v2.iter()) {
            result += f1 * f2
        }
        1.0 - result
    }
}

fn generate_random_vector() -> SillyVec {
    let mut rng = thread_rng();

    let mut foo: SillyVec = [0.0; 100];
    for i in 0..foo.len() {
        foo[i] = rng.gen();
    }

    foo
}

fn create_test_data() -> SillyComparator {
    let mut vec = Vec::new();
    for _ in 0..10000 {
        vec.push(generate_random_vector());
    }
    SillyComparator { data: vec }
}

#[bench]
fn bla(b: &mut Bencher) {
    let comparator = create_test_data();
    let vs: Vec<VectorId> = (0..10000).map(VectorId).collect();

    b.iter(|| {
        let _result: Hnsw<10, _, _> = Hnsw::generate(comparator.clone(), vs.clone(), 24, 48);
    });
}
