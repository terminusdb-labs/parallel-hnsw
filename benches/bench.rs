#![feature(test)]
extern crate test;

use parallel_hnsw::{Comparator, Layer, VectorId};
use rand::{thread_rng, Rng};
use test::Bencher;
type SillyVec = [f32; 100];
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
        let _result: Layer<10, _, _> = Layer::generate(comparator.clone(), vs.clone());
    });
}
