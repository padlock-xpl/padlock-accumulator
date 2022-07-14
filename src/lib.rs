use std::convert::TryInto;
use std::rc::Rc;
use std::collections::HashMap;

#[macro_use]
extern crate log;

#[macro_use]
extern crate serde;

use anyhow::{anyhow, Result, Context};

use thiserror::Error;

use kvdb::KeyValueDB;

pub const HASH_SIZE: usize = 32;

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

    pub fn get_proofs(&self, mut indices: Vec<usize>) -> Result<AggregatedProof> {
        // Each element takes the form, tree_index, leaf_index
        let mut indices_with_tree_index: Vec<(usize, usize)> = Vec::new();

        for index in indices.iter() {
            let tree_index = self.find_tree_that_contains_index(*index)?;

            indices_with_tree_index.push((tree_index, *index));
        }

        let mut proof_trees: HashMap<usize, ProofTree> = HashMap::new();

        for index in indices {
            let proof = self.get_proof(index).context(format!("Could not get proof for index {}", index))?;
            let leaf = self.get_leaf(index).context(format!("Could not get leaf for index {}", index))?;
            let proof_tree_index = self.find_tree_that_contains_index(index).context(format!("Could not find tree that contains the index {}", index))?;

            let mut proof_tree: ProofTree = match proof_trees.get(&proof_tree_index) {
                Some(proof_tree) => proof_tree.to_owned(),
                None => ProofTree::new(),
            };

            proof_tree.add_proof(proof, leaf).context("Could not add proof to proof tree")?;

            proof_trees.insert(proof_tree_index, proof_tree);
        }

        Ok(AggregatedProof {
            proof_trees
        })
    }

    pub fn is_aggregated_proof_valid(&self, proof: &AggregatedProof) -> Result<bool> {
        for (tree_index, proof_tree) in proof.proof_trees.iter() {
            let tree = self.trees.iter().nth(*tree_index).context(format!("tree with index {} doesn't exist", tree_index))?;

            if !proof_tree.is_valid(&tree.root())? {
                return Ok(false)
            }
        }

        Ok(true)
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

pub struct AggregatedProof {
    proof_trees: HashMap<usize, ProofTree>,
}

#[cfg(test)]
impl AggregatedProof {
    /// Will mess up the proof for testing purposes
    pub fn mess_up(&mut self) {
        for tree in self.proof_trees.values_mut() {
            tree.mess_up();
        }
    }
}

pub struct PrunedAccumulator {
    tree_roots: Vec<Hash>
}

pub type Hash = [u8; HASH_SIZE];

fn hash(data: &[u8]) -> Hash {
    blake3::hash(data).as_bytes()[0..HASH_SIZE].try_into().unwrap()
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
    use crate::*;

    use kvdb_memorydb::InMemory;
    use rand::Rng;
    
    use std::rc::Rc;

    const ACCUMULATOR_SIZE: usize = 57;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
    
    #[test]
    fn invalid_aggregated_proof_is_invalid() {
        init();
    
        let accumulator = test_accumulator();
    
        let mut proofs_to_get = Vec::new();
    
        for i in 0..15 {
            let index = rand::thread_rng().gen_range(0..(ACCUMULATOR_SIZE - 1));
    
            proofs_to_get.push(index);
        }
    
        let mut aggregated_proof = accumulator.get_proofs(proofs_to_get).unwrap();
        aggregated_proof.mess_up();
    
        assert!(accumulator.is_aggregated_proof_valid(&aggregated_proof).unwrap() == false)
    }

    fn test_accumulator() -> Accumulator<InMemory> {
        let db = Rc::new(kvdb_memorydb::create(1));
    
        let mut accumulator = Accumulator::new(db.clone());
    
        for _ in 0..ACCUMULATOR_SIZE {
            accumulator.add(rand::random(), true).unwrap();
        }
    
        accumulator
    }
}
