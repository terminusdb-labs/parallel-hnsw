use parallel_hnsw::bigvec::make_random_hnsw;
use parallel_hnsw::NodeId;

pub fn main() {
    let mut hnsw = make_random_hnsw(100000, 100);
    hnsw.improve_index(1.0, 1.0, 1.0, None);
    for ix in 0..hnsw.layer_count() {
        let layer = hnsw.get_layer_from_top(ix).unwrap();
        let supers = if ix == 0 {
            vec![layer.get_vector(NodeId(0))]
        } else {
            let super_layer = hnsw.get_layer_from_top(ix - 1).unwrap();
            super_layer.nodes.clone()
        };
        let distances = layer.node_distances(&supers);

        for (n, distance) in distances.into_iter().enumerate() {
            println!("{ix},{n},{},{}", distance.hops, distance.index_sum);
        }
    }
}
