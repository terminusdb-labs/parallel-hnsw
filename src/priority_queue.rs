use crate::OrderedFloat;

pub struct PriorityQueue<'a, Id> {
    pub data: &'a mut [Id],
    pub priorities: &'a mut [f32],
}

impl<'a, Id: PartialEq + Copy> PriorityQueue<'a, Id> {
    fn data(&'a self) -> &'a [Id] {
        self.data
    }

    fn insert_at(&mut self, idx: usize, elt: Id, priority: f32) {
        if idx < self.data.len() && self.data[idx] != elt {
            let data = &mut self.data;
            let priorities = &mut self.priorities;
            let swap_start =
                priorities.partition_point(|d| OrderedFloat(*d) != OrderedFloat(f32::MAX));

            for i in (idx + 1..swap_start + 1).rev() {
                if i == priorities.len() {
                    continue;
                }
                data[i] = data[i - 1];
                priorities[i] = priorities[i - 1];
            }
            data[idx] = elt;
            priorities[idx] = priority;
        }
    }

    pub fn insert(&mut self, elt: Id, priority: f32) {
        let idx = self
            .priorities
            .partition_point(|d| OrderedFloat(*d) < OrderedFloat(priority));
        self.insert_at(idx, elt, priority);
    }

    pub fn merge<'b>(&mut self, other_data: &'b [Id], other_priority: &'b [f32]) -> bool {
        let mut did_something = false;
        let mut last_idx = 0;
        for (other_idx, other_distance) in other_priority.iter().enumerate() {
            let i = self.priorities[last_idx..]
                .binary_search_by(|d0| OrderedFloat(*d0).cmp(&OrderedFloat(*other_distance)));
            match i {
                Ok(i) => {
                    self.insert_at(i, other_data[other_idx], *other_distance);
                    did_something = true;
                    last_idx = i;
                }
                Err(i) => {
                    if i >= self.data.len() {
                        break;
                    } else {
                        self.insert_at(i + last_idx, other_data[other_idx], *other_distance);
                        did_something = true;
                        last_idx = i;
                    }
                }
            }
        }
        did_something
    }

    fn merge_from(&mut self, other: PriorityQueue<Id>) {
        self.merge(other.data, other.priorities);
    }

    fn merge_pairs(&mut self, other: &[(Id, f32)]) {
        let (ids, priorities): (Vec<Id>, Vec<f32>) = other.iter().map(|p| *p).unzip();
        self.merge(&ids, &priorities);
    }

    pub fn iter(&'a self) -> PriorityQueueIter<'a, Id> {
        PriorityQueueIter {
            data_iter: self.data,
            priority_iter: self.priorities,
        }
    }
}

pub struct PriorityQueueIter<'iter, Id> {
    data_iter: &'iter [Id],
    priority_iter: &'iter [f32],
}

impl<Id: PartialEq + Copy> Iterator for PriorityQueueIter<'_, Id> {
    type Item = (Id, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((data_head, data_tail)) = self.data_iter.split_first() {
            if let Some((priority_head, priority_tail)) = self.priority_iter.split_first() {
                self.data_iter = data_tail;
                self.priority_iter = priority_tail;
                Some((*data_head, *priority_head))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{priority_queue::PriorityQueue, NodeId};

    #[test]
    fn fixed_length_insertion() {
        // At beginning
        let mut data = vec![NodeId(0), NodeId(3), NodeId(!0)];
        let mut priorities = vec![0.1, 1.2, 4.5];
        let mut priority_queue = PriorityQueue {
            data: &mut data,
            priorities: &mut priorities,
        };
        priority_queue.insert(NodeId(4), 0.01);
        assert_eq!(data, vec![NodeId(4), NodeId(0), NodeId(3)]);
        assert_eq!(priorities, vec![0.01, 0.1, 1.2]);

        // into empty
        let mut data = vec![NodeId(!0), NodeId(!0), NodeId(!0)];
        let mut priorities = vec![f32::MAX, f32::MAX, f32::MAX];
        let mut priority_queue = PriorityQueue {
            data: &mut data,
            priorities: &mut priorities,
        };
        priority_queue.insert(NodeId(4), 0.01);
        assert_eq!(
            data,
            vec![
                NodeId(4),
                NodeId(18446744073709551615),
                NodeId(18446744073709551615)
            ]
        );
        assert_eq!(priorities, vec![0.01, 3.4028235e38, 3.4028235e38]);

        // Don't double count
        let mut data = vec![NodeId(4), NodeId(!0), NodeId(!0)];
        let mut priorities = vec![0.01, f32::MAX, f32::MAX];
        let mut priority_queue = PriorityQueue {
            data: &mut data,
            priorities: &mut priorities,
        };
        priority_queue.insert(NodeId(4), 0.01);
        assert_eq!(
            data,
            vec![
                NodeId(4),
                NodeId(18446744073709551615),
                NodeId(18446744073709551615)
            ]
        );
        assert_eq!(priorities, vec![0.01, 3.4028235e38, 3.4028235e38]);

        // Push off the end
        let mut data = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut priorities = vec![0.1, 0.2, 0.4];
        let mut priority_queue = PriorityQueue {
            data: &mut data,
            priorities: &mut priorities,
        };
        priority_queue.insert(NodeId(4), 0.3);
        assert_eq!(data, vec![NodeId(1), NodeId(2), NodeId(4)]);
        assert_eq!(priorities, vec![0.1, 0.2, 0.3]);

        // Insert past the end
        let mut data = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut priorities = vec![0.1, 0.2, 0.3];
        let mut priority_queue = PriorityQueue {
            data: &mut data,
            priorities: &mut priorities,
        };
        priority_queue.insert(NodeId(4), 0.4);
        assert_eq!(data, vec![NodeId(1), NodeId(2), NodeId(3)]);
        assert_eq!(priorities, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn fixed_length_merge() {
        // Interleaved
        let mut data1 = vec![NodeId(0), NodeId(2), NodeId(4)];
        let mut priorities1 = vec![0.0, 0.2, 0.4];
        let mut priority_queue1 = PriorityQueue {
            data: &mut data1,
            priorities: &mut priorities1,
        };

        let mut data2 = vec![NodeId(1), NodeId(3), NodeId(5)];
        let mut priorities2 = vec![0.1, 0.3, 0.5];
        let priority_queue2 = PriorityQueue {
            data: &mut data2,
            priorities: &mut priorities2,
        };

        priority_queue1.merge_from(priority_queue2);
        assert_eq!(data1, vec![NodeId(0), NodeId(1), NodeId(2)]);
        assert_eq!(priorities1, vec![0.0, 0.1, 0.2]);
    }
}
