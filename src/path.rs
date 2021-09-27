/// A path in the tree. Represented as a list of the index to the child leaf of
/// each node
#[derive(Debug)]
pub struct Path {
    path: Vec<u8>,
    current_height: usize,
}

impl Path {
    pub fn new(height: usize, index: usize) -> Self {
        let mut path: Vec<u8> = Vec::new();

        let mut current_index = index;

        for _ in 0..height {
            path.push((current_index % 16) as u8);
            current_index /= 16;
        }

        path.reverse();

        Self {
            path,
            current_height: 0,
        }
    }
}

impl Iterator for Path {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_height >= self.path.len() {
            return None;
        }

        let item = self.path[self.current_height];
        self.current_height += 1;

        Some(item)
    }
}
