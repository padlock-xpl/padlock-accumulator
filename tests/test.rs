use padlock_accumulator::{Accumulator, Hash};

#[macro_use]
extern crate log;

use kvdb_memorydb::InMemory;

const ACCUMULATOR_SIZE: usize = 57; 

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn can_create_accumulator() {
    init();

    let mut db = kvdb_memorydb::create(1);

    let _accumulator: Accumulator<InMemory> = Accumulator::new(db);
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
    let db = kvdb_memorydb::create(1);
    let mut accumulator = Accumulator::new(db);

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
        println!("loop: {}", i);
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

    todo!()
}

#[test]
fn can_remove_hashes() {
    init();

    let mut accumulator = test_accumulator();

    for i in 0..ACCUMULATOR_SIZE {
        info!("loop: {}", i);
        let proof = accumulator.get_proof(i).unwrap();
        let leaf = accumulator.get_leaf(i).unwrap();

        accumulator.remove(&proof, i).unwrap();

        assert!(accumulator.contains_root(proof.root(leaf)) == false);
        assert!(accumulator.contains_root(proof.root(Hash::default())))
    }
}

fn test_accumulator() -> Accumulator<InMemory> {
    let db = kvdb_memorydb::create(1);

    let mut accumulator = Accumulator::new(db);

    for _ in 0..ACCUMULATOR_SIZE {
        accumulator.add(rand::random(), true).unwrap();
    }

    accumulator
}
