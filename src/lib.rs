use std::convert::TryInto;
use std::rc::Rc;

#[macro_use]
extern crate log;

use anyhow::{anyhow, Result};

use thiserror::Error;

use kvdb::KeyValueDB;

pub const HASH_SIZE: usize = 25;

mod merkle_tree;
use merkle_tree::{MerkleTree, Proof, ProofTree};

mod path;

#[derive(Debug)]
pub struct Accumulator<Db> {
    pub trees: Vec<MerkleTree<Db>>,
    num_elements: usize,
    db: Rc<Db>,
}

impl<Db: KeyValueDB> Accumulator<Db> {
    pub fn new(backing_db: Db) -> Self {
        Self {
            trees: Vec::new(),
            num_elements: 0,
            db: Rc::new(backing_db),
        }
    }

    /// Adds a hash to the accumulator. If track_proof is true, a proof of
    /// membership for this hash will be stored.
    ///
    /// Returns the index of the hash added.
    pub fn add(&mut self, hash: Hash, track_proof: bool) -> Result<usize> {
        let merkle_tree = MerkleTree::new(hash, self.db.clone())?;
        self.trees.push(merkle_tree);

        self.merge_trees()?;

        let index = self.num_elements;
        self.num_elements += 1;

        Ok(index)
    }

    fn merge_trees(&mut self) -> Result<()> {
        for i in (1..(self.trees.len())).rev() {
            let left_tree = &self.trees[i - 1];
            let right_tree = &self.trees[i];

            if left_tree.height != right_tree.height {
                break;
            }

            let right_tree = self.trees.remove(i);

            self.trees[i - 1].merge_with(right_tree)?;
            trace!("merged trees {} and {}", i - 1, i);
        }

        Ok(())
    }

    pub fn remove(&mut self, proof: &Proof, index: usize) -> Result<()> {
        let tree_index =
            self.find_tree_that_contains_index(index)?;

        trace!("tree {} contains index {}", tree_index, proof.leaf_index);

        let tree = &mut self.trees[tree_index];

        tree.remove(proof)?;

        Ok(())
    }

    pub fn get_proof(&self, index: usize) -> Result<Proof> {
        let tree_index = self.find_tree_that_contains_index(index)?;
        let index_in_tree = self.find_index_in_tree(index)?;

        let tree = &self.trees[tree_index];
        let proof = tree.get_proof(index_in_tree)?;

        Ok(proof)
    }

    pub fn get_proofs(&self, indexes: Vec<usize>) -> Result<ProofTree> {
        todo!()
    }

    pub fn get_leaf(&self, index: usize) -> Result<Hash> {
        trace!("Accumulator::get_leaf called with index {}", index);

        let tree_index = self.find_tree_that_contains_index(index)?;
        let index_in_tree = self.find_index_in_tree(index)?;

        let leaf = self
            .trees
            .iter()
            .nth(tree_index)
            .ok_or(anyhow!("leaf index {:?} isn't in the accumulator"))?
            .get_leaf(index_in_tree)?;

        Ok(leaf)
    }

    pub fn contains_root(&self, root_hash: Hash) -> bool {
        self.trees
            .iter()
            .find(|tree| tree.root() == root_hash)
            .is_some()
    }

    fn find_tree_that_contains_index(&self, index: usize) -> Result<usize> {
        let mut total_leaves = 0;

        for (i, tree) in self.trees.iter().enumerate() {
            total_leaves += tree.num_leaves();

            if index < total_leaves {
                return Ok(i);
            }
        }

        Err(anyhow!("Index doesn't exist: {}", index))
    }

    fn find_index_in_tree(&self, index: usize) -> Result<usize> {
        let mut total_leaves = 0;

        for (i, tree) in self.trees.iter().enumerate() {
            total_leaves += tree.num_leaves();

            if index < total_leaves {
                let index_in_tree = index - (total_leaves - tree.num_leaves());
                return Ok(index_in_tree);
            }
        }

        Err(anyhow!("Index doesn't exist in any tree: {}", index))
    }
}

#[derive(Debug, Clone, Copy)]
struct Root {
    hash: Hash,
    height: usize,
}

impl Root {
    fn new(hash: Hash) -> Self {
        Self { hash, height: 1 }
    }

    fn merge_with(&mut self, root: Self) {
        if self.height != root.height {
            panic!("Cannot merge roots of different heights");
        }

        let to_hash = [self.hash, root.hash].concat();
        self.hash = hash(&to_hash);
        self.height += 1;
    }
}

pub type Hash = [u8; HASH_SIZE];

fn hash(data: &[u8]) -> Hash {
    blake3::hash(data).as_bytes()[0..25].try_into().unwrap()
}

#[derive(Debug, Error)]
pub enum AccumulatorError {
    #[error("Index doesn't exist {}", .0)]
    IndexDoesntExist(usize),
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error>),
}

#[cfg(test)]
mod tests {
    use crate::Root;

    pub fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn can_create_root() {
        init();

        let _root = Root::new(rand::random());
    }

    #[test]
    fn can_merge_roots() {
        init();

        let mut root1 = Root::new(rand::random());
        let root2 = Root::new(rand::random());

        root1.merge_with(root2);
    }
}
