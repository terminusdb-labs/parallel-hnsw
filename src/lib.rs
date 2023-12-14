#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy)]
struct VectorId(usize);
#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy)]
struct NodeId(usize);

pub struct Layer<const M: usize> {
    metric: Box<Fn(VectorId, VectorId) -> f32>,
    nodes: Vec<VectorId>,
    neighbours: Vec<NodeId>,
}

impl<const M: usize> Layer<M> {
    fn get_node(&self, v: VectorId) -> NodeId {
        NodeId(self.nodes.binary_search(&v).unwrap())
    }

    fn get_vector(&self, n: NodeId) -> VectorId {
        self.nodes[n.0]
    }

    fn get_neighbours(&self, n: NodeId) -> &[NodeId] {
        &self.neighbours[(n.0 * M)..((n.0 + 1) * M)]
    }

    fn generate(vs: Vec<VectorId>) -> Self {
        //
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
