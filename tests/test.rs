use padlock_accumulator::{Accumulator, Hash};

#[macro_use]
extern crate log;

use kvdb_memorydb::InMemory;
use rand::{Rng, SeedableRng};

use std::rc::Rc;

const ACCUMULATOR_SIZE: usize = 57;

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn can_create_accumulator() {
    init();

    let mut db = Rc::new(kvdb_memorydb::create(1));

    let _accumulator: Accumulator<InMemory> = Accumulator::new(db.clone());
}

#[test]
fn can_add_hashes() {
    init();

    let accumulator = test_accumulator();
    for tree in accumulator.trees {
        println!("{:?}", tree.root());
    }
}

#[test]
fn get_leaf_returns_correct_leaf() {
    let mut leafs = Vec::new();
    let db = Rc::new(kvdb_memorydb::create(1));
    let mut accumulator = Accumulator::new(db.clone());

    for _ in 0..ACCUMULATOR_SIZE {
        let hash = rand::random();

        leafs.push(hash);
        accumulator.add(hash, true).unwrap();
    }

    for i in 0..ACCUMULATOR_SIZE {
        let leaf = accumulator.get_leaf(i).unwrap();
        assert!(leaf == leafs[i])
    }
}

#[test]
fn can_create_valid_proofs() {
    init();

    let accumulator = test_accumulator();
    for i in 0..ACCUMULATOR_SIZE {
        let proof = accumulator.get_proof(i).unwrap();
        let leaf = accumulator.get_leaf(i).unwrap();
        assert!(accumulator.contains_root(proof.root(leaf)));
    }
}

#[test]
fn invalid_proof_is_recognized() {
    init();

    let accumulator = test_accumulator();

    for i in 0..ACCUMULATOR_SIZE {
        let proof = accumulator.get_proof(i).unwrap();
        let leaf = accumulator.get_leaf(i).unwrap();
        println!("loop: {}", i);
        assert!(accumulator.contains_root(proof.root(rand::random())) == false);
    }
}

#[test]
fn can_create_valid_aggregated_proof() {
    init();

    let accumulator = test_accumulator();

    let mut proofs_to_get = Vec::new();

    for i in 0..15 {
        let index = rand::thread_rng().gen_range(0..(ACCUMULATOR_SIZE - 1));

        proofs_to_get.push(index);
    }

    let aggregated_proof = accumulator.get_proofs(proofs_to_get).unwrap();

    assert!(accumulator.is_aggregated_proof_valid(&aggregated_proof).unwrap());
}

#[test]
fn can_remove_hashes() {
    init();

    let mut accumulator = test_accumulator();

    for i in 0..2 {
        info!("loop: {}", i);
        let proof = accumulator.get_proof(i).unwrap();
        let leaf = accumulator.get_leaf(i).unwrap();

        accumulator.remove(&proof, i).unwrap();

        assert!(accumulator.contains_root(proof.root(leaf)) == false);

        info!("proof root: {:?}", proof.root(Hash::default()));
        assert!(accumulator.contains_root(proof.root(Hash::default())));
    }
}

#[test]
fn accumulator_works_after_recovering_from_db() {
    let accumulator = test_accumulator();
    accumulator.save_to_db().unwrap();

    let db = accumulator.db();
    let accumulator = Accumulator::from_db(db).unwrap();

    accumulator.get_proof(ACCUMULATOR_SIZE - 1).unwrap();
    accumulator.get_leaf(2).unwrap();
}

fn test_accumulator() -> Accumulator<InMemory> {
    let db = Rc::new(kvdb_memorydb::create(1));

    // Uses a seeded random number generator to make sure the tree is random, but the same every
    // time, to make debugging easier
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(10);

    let mut accumulator = Accumulator::new(db.clone());

    for _ in 0..ACCUMULATOR_SIZE {
        accumulator.add(rng.gen(), true).unwrap();
    }

    accumulator
}
