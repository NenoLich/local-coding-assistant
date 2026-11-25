// Small JavaScript file for benchmarking

function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

class SimpleClass {
    constructor(value) {
        this.value = value;
    }

    increment() {
        this.value += 1;
    }

    getValue() {
        return this.value;
    }
}

function main() {
    const obj = new SimpleClass(10);
    obj.increment();
    const result = add(obj.getValue(), 5);
    console.log(result);
}

main();
