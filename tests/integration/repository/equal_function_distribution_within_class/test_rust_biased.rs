struct TestRustStruct;

impl Default for TestRustStruct {
    fn default() -> Self {
        Self
    }
}

impl TestRustStruct {
    fn biased(&self) {
    }

    fn unbiased_1(&self) {
        self.biased();
    }
    fn unbiased_2(&self) {
        self.biased();
    }
    fn unbiased_3(&self) {
        self.biased();
    }
}

struct OutsiderRustStruct;

impl OutsiderRustStruct {
    fn unbiased_1(&self) {
        let test_struct = TestRustStruct;
        test_struct.biased();
    }
    fn unbiased_2(&self) {
        let test_struct = TestRustStruct;
        test_struct.biased();
    }
    fn unbiased_3(&self) {
        let test_struct = TestRustStruct;
        test_struct.biased();
    }
}