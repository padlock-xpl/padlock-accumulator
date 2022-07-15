use std::rc::Rc;
use std::collections::HashMap;

use crate::{hash, path::Path, Hash};

use anyhow::{anyhow, Result, Context};

use kvdb::KeyValueDB;
use serde::{Deserialize, Serialize};

#[cfg(test)]
use rand::Rng;

/// A merkle tree that stores data on disk.
#[derive(Debug)]
pub struct MerkleTree<Db> {
    db: Rc<Db>,
    root_node_hash: Hash,
    node_height: usize,
    pub height: usize,
}

impl<Db: KeyValueDB> MerkleTree<Db> {
    pub fn new(leaf_hash: Hash, db: Rc<Db>) -> Result<Self> {
        trace!("called MerkleTree::new with leaf_hash: {:?}", &leaf_hash);

        let node = Node::new(leaf_hash, true);
        let mut transaction = db.transaction();
        transaction.put(0, &leaf_hash, &bincode::serialize(&node)?);
        db.write(transaction)?;

        Ok(Self {
            db,
            root_node_hash: node.root()?,
            node_height: 1,
            height: 1,
        })
    }

    pub fn from_parts(db: Rc<Db>, root_node_hash: Hash, node_height: usize, height: usize) -> Self {
        Self {
            db,
            root_node_hash,
            node_height,
            height,
        }
    }

    pub fn num_leaves(&self) -> usize {
        2usize.pow(self.height as u32 - 1)
    }

    pub fn merge_with(&mut self, tree: MerkleTree<Db>) -> Result<()> {
        if self.height != tree.height {
            return Err(anyhow!("Can't merge trees of different heights"));
        }

        let mut transaction = self.db.transaction();

        let mut root_node = self.root_node()?;
        transaction.delete(0, &root_node.root()?);

        let new_root_node = root_node.merge_with(tree.root_node()?)?;

        if let Some(new_root_node) = new_root_node {
            self.node_height += 1;

            let new_root_node_bytes = bincode::serialize(&new_root_node)?;
            let root_node_bytes = bincode::serialize(&root_node)?;

            transaction.put(0, &new_root_node.root()?, &new_root_node_bytes);
            transaction.put(0, &root_node.root()?, &root_node_bytes);

            self.root_node_hash = new_root_node.root()?;
        } else {
            let root_node_bytes = bincode::serialize(&root_node)?;
            transaction.put(0, &root_node.root()?, &root_node_bytes);

            self.root_node_hash = root_node.root()?;
        }

        self.db.write(transaction)?;

        self.height += 1;

        Ok(())
    }

    pub fn get_proof(&self, index: usize) -> Result<Proof> {
        trace!("merkle_tree::get_proof called with index {}", index);

        let path = Path::new(self.node_height, index);
        trace!("created path: {:?}", path);

        let mut current_node: Node;
        let mut next_node_hash = self.root();
        let mut proofs = Vec::new();

        for index in path {
            let current_node_bytes = self.db.get(0, &next_node_hash)?.ok_or(anyhow!(
                "Node {:?} doesn't exist in database",
                next_node_hash
            ))?;

            current_node = bincode::deserialize(&current_node_bytes)?;

            let proof = current_node.get_proof(index as usize)?;

            // assert!(proof.root(*current_node.get_leaf(index as usize)?) == current_node.root());

            proofs.push(proof);
            next_node_hash = current_node.get_leaf(index as usize)?;

            if current_node.is_leaf {
                break;
            }
        }

        debug!("leaf hash in proof: {:?}", next_node_hash);

        proofs.reverse();

        // concatenate proofs
        for _ in 0..proofs.len() - 1 {
            let mut next_proof = proofs.remove(1);
            proofs[0].concat(&mut next_proof);
        }

        debug!("constructed proof: {:#?}", proofs[0]);

        proofs[0].leaf_index = index;

        Ok(proofs[0].clone())
    }

    pub fn get_leaf(&self, index: usize) -> Result<Hash> {
        trace!("merkle_tree::get_leaf called with index {}", index);

        let path = Path::new(self.node_height, index);

        let mut current_node: Node;
        let mut next_node_hash = self.root();

        for index in path {
            let current_node_bytes = self.db.get(0, &next_node_hash)?.ok_or(anyhow!(
                "Node {:?} doesn't exist in database",
                next_node_hash
            ))?;

            current_node = bincode::deserialize(&current_node_bytes)?;

            next_node_hash = current_node.get_leaf(index as usize)?;

            if current_node.is_leaf {
                break;
            }
        }

        Ok(next_node_hash)
    }

    pub fn remove(&mut self, proof: &Proof) -> Result<()> {
        trace!(
            "MerkleTree::remove called with proof.leaf_index {}",
            proof.leaf_index
        );

        trace!(
            "Removing index {} from tree with root {:?}",
            proof.leaf_index,
            self.root()
        );

        // add all relevant nodes to array
        let path = Path::new(self.node_height, proof.leaf_index);

        let mut current_node: Node;
        let mut next_node_hash = self.root();

        let mut nodes_and_parent_indexes = Vec::new();

        for index in path {
            let current_node_bytes = self.db.get(0, &next_node_hash)?.ok_or(anyhow!(
                "Node {:?} doesn't exist in database",
                next_node_hash
            ))?;

            current_node = bincode::deserialize(&current_node_bytes)?;
            nodes_and_parent_indexes.push(current_node.clone());

            next_node_hash = current_node.get_leaf(index as usize)?;

            if current_node.is_leaf {
                break;
            }
        }

        // remove all nodes from database
        let mut transaction = self.db.transaction();

        for node in &nodes_and_parent_indexes {
            transaction.delete(0, &node.root()?);
        }

        // now there is a list of relevant nodes ordered from highest to lowest
        // set leaf in leaf node to zeros
        let leaf = &mut nodes_and_parent_indexes.last_mut().ok_or(anyhow!(
            "Cannot delete index {} because no nodes were found on it's path",
            proof.leaf_index
        ))?;

        let mut last_node_pre_root = leaf.root()?;

        leaf.prune_leaf(proof.leaf_index % 16);

        let mut last_node_root = leaf.root()?;

        if last_node_root == last_node_pre_root {
            return Err(anyhow!("Removing from merkle tree is not functioning properly; please make issue on this crate's github"));
        }

        // set leaf in every node to new hashes
        for node_and_parent_index in nodes_and_parent_indexes.iter_mut().rev().skip(1) {
            let last_node_index = node_and_parent_index.get_index_from_hash(last_node_pre_root)?;
            last_node_pre_root = node_and_parent_index.root()?;
            node_and_parent_index.set_leaf(last_node_index, last_node_root);
            last_node_root = node_and_parent_index.root()?;
        }

        // add all nodes back to database
        for node in &nodes_and_parent_indexes {
            transaction.put(0, &node.root()?, &bincode::serialize(&node)?);
        }

        self.db.write(transaction)?;

        self.root_node_hash = nodes_and_parent_indexes
            .first()
            .ok_or(anyhow!(
                "No nodes were found in index {}'s path",
                proof.leaf_index
            ))?
            .root()?;

        trace!("New tree root: {:?}", self.root());

        Ok(())
    }

    fn root_node(&self) -> Result<Node> {
        let node_bytes = self
            .db
            .get(0, &self.root_node_hash)?
            .ok_or(anyhow!("Root node hash not in database"))?;
        let node = bincode::deserialize(&node_bytes)?;
        Ok(node)
    }

    pub fn root(&self) -> Hash {
        self.root_node_hash
    }
}

/// A merkle tree in it's intermediate form before being serialized or
/// deserialized.
#[derive(Serialize, Deserialize)]
pub struct MerkleTreeForSerialization {
    root_node_hash: Hash,
    node_height: usize,
    height: usize,
}

impl MerkleTreeForSerialization {
    fn from_merkle_tree<Db: KeyValueDB>(merkle_tree: &MerkleTree<Db>) -> Self {
        Self {
            root_node_hash: merkle_tree.root_node_hash,
            node_height: merkle_tree.node_height,
            height: merkle_tree.height,
        }
    }

    pub fn from_merkle_trees<Db: KeyValueDB>(merkle_trees: &[MerkleTree<Db>]) -> Vec<Self> {
        let mut merkle_tree_for_serialization_vec = Vec::new();

        for merkle_tree in merkle_trees.iter() {
            let merkle_tree_for_serialization =
                MerkleTreeForSerialization::from_merkle_tree(&merkle_tree);

            merkle_tree_for_serialization_vec.push(merkle_tree_for_serialization);
        }

        merkle_tree_for_serialization_vec
    }

    fn to_merkle_tree<Db: KeyValueDB>(&self, db: Rc<Db>) -> MerkleTree<Db> {
        MerkleTree {
            db,
            root_node_hash: self.root_node_hash,
            node_height: self.node_height,
            height: self.height,
        }
    }

    pub fn vec_to_merkle_tree<Db: KeyValueDB>(vec: Vec<Self>, db: Rc<Db>) -> Vec<MerkleTree<Db>> {
        let mut merkle_tree_vec = Vec::new();

        for merkle_tree_for_serialization in vec.iter() {
            merkle_tree_vec.push(merkle_tree_for_serialization.to_merkle_tree(db.clone()));
        }

        merkle_tree_vec
    }
}

/// The merkle tree is split into many smaller merkle trees. Each sub tree is
/// called a node. These smaller merkle trees don't actually store a full merkle
/// tree, but rather it stores just the leaves. In order to retrieve a proof, it
/// concatenates a proof from every node in the branch. Proofs are retrieved
/// from nodes by recreating the subtree based off of the leaves, getting the
/// proof, then discarding the subtree (except the leaves of course)
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Node {
    is_leaf: bool,
    leaves: HashMap<usize, Hash>,
    num_leaves: usize,
}

impl Node {
    fn new(leaf_hash: Hash, is_leaf: bool) -> Self {
        let mut leaves = HashMap::new();
        leaves.insert(0, leaf_hash);

        Self {
            is_leaf,
            leaves,
            num_leaves: 1,
        }
    }

    fn root(&self) -> Result<Hash> {
        let tree = self.build_tree()?;
        Ok(tree.last().unwrap().clone())
    }

    fn get_leaf(&self, index: usize) -> Result<Hash> {
        if index >= self.num_leaves {
            return Err(anyhow!("Index cannot be greater than the number of leaves"))
        }

        let leaf = match self.leaves.get(&index) {
            Some(leaf) => *leaf,
            None => Hash::default(),
        };

        Ok(leaf)
    }

    fn set_leaf(&mut self, index: usize, leaf: Hash) {

        match self.leaves.insert(index, leaf) {
            Some(_) => {}
            None => { self.num_leaves += 1 }
        }
    }

    // Effectively sets leaf to zeros. Leaves that are zeros aren't actually stored in order to
    // save space.
    fn prune_leaf(&mut self, index: usize) {
        self.leaves.remove(&index);
    }

    fn get_index_from_hash(&self, hash: Hash) -> Result<usize> {
        for (index, leaf) in self.leaves.iter() {
            if *leaf == hash {
                return Ok(*index)
            }
        }

        Err(anyhow!("Could not get index from hash {:?} in node {:?}", hash, self.root()))
    }

    fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    fn leaves_as_slice(&self) -> Result<Vec<Hash>> {
        let mut vec_of_leaves = Vec::new();

        for index in 0..self.num_leaves {
            vec_of_leaves.push(self.get_leaf(index)?)
        }

        Ok(vec_of_leaves)
    }

    /// Returns a merkle tree from self.leaves, represented as
    /// [0, 1, 2, 3, h(0, 1), h(2, 3), h(h(0, 1) h(2, 3))]
    /// The last value in the vector being the root.
    fn build_tree(&self) -> Result<Vec<Hash>> {
        if self.leaves.len() == 1 {
            return Ok(vec![self.get_leaf(0)?]);
        }

        let mut tree: Vec<Hash> = self.leaves_as_slice()?.into();

        let mut i = 0;

        loop {
            if i >= tree.len() - 1 {
                break;
            }

            let left = tree[i];
            let right = tree[i + 1];

            let to_hash = [left, right].concat();
            let hash = hash(&to_hash);

            tree.push(hash);

            i += 2;
        }

        Ok(tree)
    }

    fn append_leaf(&mut self, leaf: Hash) {
        self.set_leaf(self.num_leaves, leaf)
    }

    fn append_leaves(&mut self, leaves: &[Hash]) {
        for leaf in leaves.iter() {
            self.append_leaf(*leaf);
        }
    }

    pub fn height(&self) -> usize {
        ((self.leaves.len() as f64).log2() + 1f64) as usize
    }

    /// Merges two nodes together, and returns another node if the merged node
    /// has more than 16 leaves
    fn merge_with(&mut self, mut node: Node) -> Result<Option<Node>> {
        if self.num_leaves != node.num_leaves {
            return Err(anyhow!("Cannot merge two nodes of different heights"));
        }

        if self.num_leaves == 16 {
            //let new_leaves = vec![self.root(), node.root()];
            let mut new_leaves = HashMap::new();
            new_leaves.insert(0, self.root()?);
            new_leaves.insert(1, node.root()?);

            return Ok(Some(Self {
                is_leaf: false,
                leaves: new_leaves,
                num_leaves: 2
            }));
        }

        self.append_leaves(&mut node.leaves_as_slice()?);

        Ok(None)
    }

    fn get_proof(&self, index: usize) -> Result<Proof> {
        let tree = self.build_tree()?;

        let mut proof = Vec::new();

        let mut layer_start = 0;
        let mut current_index = index;
        let mut current_layer_len = self.num_leaves;

        while current_layer_len > 1 {
            let sibling_index = match current_index % 2 == 0 {
                true => current_index + 1,
                false => current_index - 1,
            };

            proof.push(tree[layer_start + sibling_index]);

            layer_start += current_layer_len;

            current_index /= 2;

            current_layer_len /= 2;
        }

        Ok(Proof {
            hashes: proof,
            leaf_index: index,
        })
    }
}

/// A sparse merkle tree that contains multiple proofs of memberships.
#[derive(Debug, Clone)]
pub struct ProofTree {
    hash_map: HashMap<TreeKey, Hash>,
    /// Height will be None if no proofs have been added to the tree
    height: Option<usize>,
}

impl ProofTree {
    pub fn new() -> Self {
        Self {
            hash_map: HashMap::new(),
            height: None
        }
    }

    pub fn root(&self) -> Option<&Hash> {
        let key = TreeKey { index: 0, height: 0 };
        self.hash_map.get(&key)
    }

    fn add_hash(&mut self, key: TreeKey, hash: Hash) {
        trace!("adding hash {:?} with key {:?}", hash, key);
        self.hash_map.insert(key, hash);
    }

    fn get_hash(&self, key: &TreeKey) -> Option<&Hash> {
        self.hash_map.get(key)
    }

    pub fn add_proof(&mut self, proof: Proof, proof_leaf_hash: Hash) -> Result<()> {
        trace!("add_proof called");

        if let Some(self_root) = self.root() {
            if proof.root(proof_leaf_hash) != *self_root {
                return Err(anyhow!("Proof must have the same root hash as the proof tree"))
            }
        }

        let mut height = proof.height_of_tree();
        if let Some(current_height) = self.height {
            if current_height != height {
                return Err(anyhow!("New proof must be the same height as the current proof tree"));
            }
        } else {
            self.height = Some(height)
        }

        let layers = proof.layers(&proof_leaf_hash);

        for layer in layers {
            let left_key = TreeKey { index: layer.left_index, height };
            self.add_hash(left_key, layer.left);

            let right_key = TreeKey { index: layer.left_index + 1, height };
            self.add_hash(right_key, layer.right);

            height -= 1;
        }
        
        let root_key = TreeKey { index: 0, height: 0 };
        self.add_hash(root_key, proof.root(proof_leaf_hash));

        Ok(())
    }

    /// Checks whether the proof tree is valid (all node children hash to their parent's value) and
    /// that the proof tree root is equal to the supplied root
    pub fn is_valid(&self, root_of_tree_to_check_against: &Hash) -> Result<bool> {
        let root_key = TreeKey { index: 0, height: 0 };
        let root_hash = self.get_hash(&root_key).context("Proof tree doesn't have a root node")?;
        
        if root_hash != root_of_tree_to_check_against {
            return Ok(false)
        }

        let is_valid = self.check_node(&root_key)?;

        Ok(is_valid)
    }

    /// Checks whether a node's children hashes to their the nodes value. Also checks if both
    /// children are valid. Therefore, calling this on just the root node will check every existing
    /// node in the proof tree.
    fn check_node(&self, hash_key: &TreeKey) -> Result<bool> {
        let tree_height = self.height.context("Tree doesn't have a set height yet (you need to add proofs first)")?;
        
        trace!("Checking node with key {:?}", hash_key);
        
        // Checking whether this hash is a leaf hash
        if hash_key.height >= tree_height {
            return Ok(true)
        }

        let left_child_key = TreeKey {
            index: hash_key.index * 2,
            height: hash_key.height + 1,
        };

        let left_hash = match self.get_hash(&left_child_key) {
            Some(left_hash) => left_hash,
            None => {
                return Ok(true)
            }
        };
        
        if !self.check_node(&left_child_key)? {
            return Ok(false)
        }

        let right_child_key = TreeKey {
            index: left_child_key.index + 1,
            height: left_child_key.height,
        };

        let right_hash = match self.get_hash(&right_child_key) {
            Some(left_hash) => left_hash,
            None => {
                return Ok(true)
            }
        };

        if !self.check_node(&right_child_key)? {
            return Ok(false)
        }

        let data_to_hash = [*left_hash, *right_hash].concat();
        let left_and_right_hashed = hash(&data_to_hash);
        let node_hash = self.get_hash(hash_key).context(format!("node with key {:?} doesn't exist", hash_key))?;

        if left_and_right_hashed != *node_hash {
            return Ok(false)
        }

        trace!("node with key {:?} is valid", hash_key);

        Ok(true)
    }

    fn contains_leaf(&self, leaf: Hash, leaf_index: usize) -> bool {
        let height = match self.height {
            Some(height) => height,
            None => { return false }
        };

        let key = TreeKey { index: leaf_index, height };
        if let Some(leaf_in_tree) = self.get_hash(&key) {
            if leaf == *leaf_in_tree {
                return true
            }
        }

        false
    }

    /// Will mess up the proof for testing purposes. Should not be called unless using for test
    ///
    /// Chooses four random hashes to mess up
    #[cfg(test)]
    pub fn mess_up(&mut self) {
        for _ in 0..4 {
            let index: usize = rand::thread_rng().gen_range(0..self.hash_map.len());
            let random: [u8; 32] = rand::random();
            self.hash_map.iter_mut().nth(index).unwrap().1.copy_from_slice(&random);
        }
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct TreeKey {
    index: usize,
    height: usize,
}

#[derive(Debug, Clone)]
pub struct Proof {
    hashes: Vec<Hash>,
    pub leaf_index: usize,
}

impl Proof {
    pub fn new(hashes: Vec<Hash>, leaf_index: usize) -> Self {
        Self { hashes, leaf_index }
    }

    pub fn root(&self, leaf_hash: Hash) -> Hash {
        let mut prev_hash = leaf_hash;
        let mut current_index = self.leaf_index;

        for current_hash in self.hashes.iter() {
            let to_hash = match current_index % 2 == 0 {
                true => [prev_hash, *current_hash].concat(),
                false => [*current_hash, prev_hash].concat(),
            };

            prev_hash = hash(&to_hash);
            current_index /= 2;
        }

        prev_hash
    }

    pub fn height_of_tree(&self) -> usize {
        self.hashes.len()
    }

    /// Adds a proof onto the end of the proof. Used to combine proofs between
    /// different subtrees.
    pub fn concat(&mut self, proof: &mut Self) {
        self.hashes.append(&mut proof.hashes);
    }

    /// Returns a vector of layers, each layer being the two hashes that are hashed to form the
    /// next layer
    fn layers(&self, leaf_hash: &Hash) -> Vec<ProofLayer> {
        let mut prev_hash = *leaf_hash;
        let mut current_index = self.leaf_index;

        let mut layers: Vec<ProofLayer> = Vec::new();

        for current_hash in self.hashes.iter() {
            let to_hash = match current_index % 2 == 0 {
                true => {
                    let layer = ProofLayer {
                        left: prev_hash,
                        left_index: current_index,
                        right: *current_hash,
                    };

                    layers.push(layer);
                    [prev_hash, *current_hash].concat()
                },
                false => {
                    let layer = ProofLayer {
                        left: *current_hash,
                        left_index: current_index - 1,
                        right: prev_hash,
                    };

                    layers.push(layer);
                    [*current_hash, prev_hash].concat()
                }
            };

            prev_hash = hash(&to_hash);
            current_index /= 2;
        }

        layers
    }
}

struct ProofLayer {
    left: Hash,
    left_index: usize,
    right: Hash,
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::merkle_tree::*;

    pub fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn can_create_merkle_tree() {
        init();

        let _tree = create_large_tree();
    }

    #[test]
    fn can_merge_merkle_trees() {
        init();

        let db = Rc::new(kvdb_memorydb::create(1));

        let mut tree1 = MerkleTree::new(rand::random(), db.clone()).unwrap();
        let tree2 = MerkleTree::new(rand::random(), db.clone()).unwrap();

        tree1.merge_with(tree2).unwrap();
    }

    #[test]
    fn can_get_root() {
        init();

        let tree = create_large_tree();

        let _root = tree.root();
    }

    #[test]
    fn can_create_valid_proof() {
        init();

        let tree = create_large_tree();

        debug!(
            "root node subtree: {:#?}",
            tree.root_node().unwrap().build_tree()
        );

        for i in 0..4 {
            debug!("iteration: {}", i);

            let proof = tree.get_proof(i).unwrap();
            let leaf_hash = tree.get_leaf(i).unwrap();

            debug!("leaf hash: {:?}", leaf_hash);

            debug!(
                "proof root: {:#?}\ntree root: {:#?}",
                &proof.root(leaf_hash),
                tree.root()
            );

            assert!(proof.root(leaf_hash) == tree.root())
        }
    }

    #[test]
    fn can_create_valid_aggregated_proof_unit() {
        init();

        let tree = create_large_tree();
        let proof_one = tree.get_proof(0).unwrap();
        let leaf_one = tree.get_leaf(0).unwrap();
        let proof_two = tree.get_proof(1).unwrap();
        let leaf_two = tree.get_leaf(1).unwrap();

        let mut aggregated_proof = ProofTree::new();

        aggregated_proof.add_proof(proof_one, leaf_one).unwrap();
        aggregated_proof.add_proof(proof_two, leaf_two).unwrap();

        assert!(aggregated_proof.is_valid(&tree.root()).unwrap());
        assert!(aggregated_proof.contains_leaf(leaf_one, 0));
        assert!(aggregated_proof.contains_leaf(leaf_two, 1));
    }

    #[test]
    fn invalid_aggregated_proof_is_invalid_unit() {
        init();

        let tree = create_large_tree();
        let proof_one = tree.get_proof(0).unwrap();
        let leaf_one = tree.get_leaf(0).unwrap();

        let mut proof_two = tree.get_proof(1).unwrap();
        let leaf_two = tree.get_leaf(1).unwrap();

        let mut aggregated_proof = ProofTree::new();

        aggregated_proof.add_proof(proof_one, leaf_one).unwrap();
        aggregated_proof.add_proof(proof_two, leaf_two).unwrap();

        aggregated_proof.mess_up();

        assert!(aggregated_proof.is_valid(&tree.root()).unwrap() == false)
    }

    #[test]
    fn can_remove_leaf() {
        init();

        for i in 0..4 {
            let mut tree = create_large_tree();
            let pre_root = tree.root();

            let proof = tree.get_proof(i).unwrap();

            tree.remove(&proof).unwrap();

            assert!(pre_root != tree.root())
        }
    }

    fn create_large_tree() -> MerkleTree<kvdb_memorydb::InMemory> {
        let db = Rc::new(kvdb_memorydb::create(1));

        let mut tree1 = MerkleTree::new(rand::random(), db.clone()).unwrap();
        let tree2 = MerkleTree::new(rand::random(), db.clone()).unwrap();
        let mut tree3 = MerkleTree::new(rand::random(), db.clone()).unwrap();
        let tree4 = MerkleTree::new(rand::random(), db.clone()).unwrap();

        tree1.merge_with(tree2).unwrap();
        tree3.merge_with(tree4).unwrap();

        tree1.merge_with(tree3).unwrap();

        tree1
    }
}
