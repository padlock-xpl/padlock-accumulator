use std::convert::TryInto;
use std::rc::Rc;

#[macro_use]
extern crate log;

#[macro_use]
extern crate serde;

use anyhow::{anyhow, Result};

use thiserror::Error;

use kvdb::KeyValueDB;

pub const HASH_SIZE: usize = 25;

mod merkle_tree;
use merkle_tree::{MerkleTree, MerkleTreeForSerialization, Proof, ProofTree};

mod path;

#[derive(Debug)]
pub struct Accumulator<Db> {
    pub trees: Vec<MerkleTree<Db>>,
    num_elements: usize,
    db: Rc<Db>,
}

impl<Db: KeyValueDB> Accumulator<Db> {
    pub fn new(backing_db: Rc<Db>) -> Self {
        Self {
            trees: Vec::new(),
            num_elements: 0,
            db: backing_db,
        }
    }

    /// Consumes self, and returns the database
    pub fn db(self) -> Rc<Db> {
        self.db
    }

    pub fn save_to_db(&self) -> Result<()> {
        let accumulator_info = AccumulatorInfo::new(&self);

        let mut transaction = self.db.transaction();
        transaction.put(
            0,
            b"accumulator_info",
            &bincode::serialize(&accumulator_info)?,
        );
        self.db.write(transaction)?;

        Ok(())
    }

    pub fn from_db(db: Rc<Db>) -> Result<Self> {
        let accumulator_info: AccumulatorInfo = match db.get(0, b"accumulator_info")? {
            Some(accumulator_info) => bincode::deserialize(&accumulator_info)?,
            None => return Err(anyhow!("accumulator_info doesn't exist in database")),
        };

        let trees =
            MerkleTreeForSerialization::vec_to_merkle_tree(accumulator_info.trees, db.clone());

        Ok(Self {
            trees,
            num_elements: accumulator_info.num_elements,
            db,
        })
    }

    /// Adds a hash to the accumulator. If track_proof is true, a proof of
    /// membership for this hash will be stored.
    ///
    /// TODO add support for tracking only certain proofs as opposed to all
    /// proofs
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
        let tree_index = self.find_tree_that_contains_index(index)?;

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

/// Used for saving the accumulator to a database
#[derive(Serialize, Deserialize)]
pub struct AccumulatorInfo {
    trees: Vec<MerkleTreeForSerialization>,
    num_elements: usize,
}

impl AccumulatorInfo {
    fn new<Db: KeyValueDB>(accumulator: &Accumulator<Db>) -> Self {
        Self {
            trees: MerkleTreeForSerialization::from_merkle_trees(&accumulator.trees),
            num_elements: accumulator.num_elements,
        }
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
