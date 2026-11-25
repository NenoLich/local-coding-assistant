// Medium JavaScript file for benchmarking with more complex structure

class OperationType {
    static ADD = 'add';
    static SUBTRACT = 'subtract';
    static MULTIPLY = 'multiply';
    static DIVIDE = 'divide';
}

class OperationResult {
    constructor(value, operation, success, errorMessage = null) {
        this.value = value;
        this.operation = operation;
        this.success = success;
        this.errorMessage = errorMessage;
    }
}

class Calculator {
    constructor() {
        this.history = [];
        this.currentValue = 0;
    }

    add(x) {
        try {
            this.currentValue += x;
            const result = new OperationResult(
                this.currentValue,
                OperationType.ADD,
                true
            );
            this.history.push(result);
            return result;
        } catch (e) {
            return new OperationResult(
                0,
                OperationType.ADD,
                false,
                e.message
            );
        }
    }

    subtract(x) {
        try {
            this.currentValue -= x;
            const result = new OperationResult(
                this.currentValue,
                OperationType.SUBTRACT,
                true
            );
            this.history.push(result);
            return result;
        } catch (e) {
            return new OperationResult(
                0,
                OperationType.SUBTRACT,
                false,
                e.message
            );
        }
    }

    multiply(x) {
        try {
            this.currentValue *= x;
            const result = new OperationResult(
                this.currentValue,
                OperationType.MULTIPLY,
                true
            );
            this.history.push(result);
            return result;
        } catch (e) {
            return new OperationResult(
                0,
                OperationType.MULTIPLY,
                false,
                e.message
            );
        }
    }

    divide(x) {
        try {
            if (x === 0) {
                throw new Error("Cannot divide by zero");
            }
            this.currentValue /= x;
            const result = new OperationResult(
                this.currentValue,
                OperationType.DIVIDE,
                true
            );
            this.history.push(result);
            return result;
        } catch (e) {
            return new OperationResult(
                0,
                OperationType.DIVIDE,
                false,
                e.message
            );
        }
    }

    getHistory() {
        return [...this.history];
    }

    clearHistory() {
        this.history = [];
    }

    reset() {
        this.currentValue = 0;
        this.clearHistory();
    }
}

class ScientificCalculator extends Calculator {
    power(exponent) {
        try {
            this.currentValue **= exponent;
            const result = new OperationResult(
                this.currentValue,
                OperationType.MULTIPLY,
                true
            );
            this.history.push(result);
            return result;
        } catch (e) {
            return new OperationResult(
                0,
                OperationType.MULTIPLY,
                false,
                e.message
            );
        }
    }

    squareRoot() {
        try {
            if (this.currentValue < 0) {
                throw new Error("Cannot calculate square root of negative number");
            }
            this.currentValue **= 0.5;
            const result = new OperationResult(
                this.currentValue,
                OperationType.MULTIPLY,
                true
            );
            this.history.push(result);
            return result;
        } catch (e) {
            return new OperationResult(
                0,
                OperationType.MULTIPLY,
                false,
                e.message
            );
        }
    }
}

function performCalculationSequence(calc) {
    calc.add(100);
    calc.subtract(25);
    calc.multiply(2);
    calc.divide(3);
    return calc.currentValue;
}

function main() {
    const calc = new ScientificCalculator();
    
    // Perform basic operations
    calc.add(50);
    calc.subtract(10);
    calc.multiply(3);
    calc.divide(4);
    
    // Perform scientific operations
    calc.power(2);
    calc.squareRoot();
    
    // Perform calculation sequence
    const result = performCalculationSequence(calc);
    
    console.log(`Final result: ${result}`);
    console.log(`History length: ${calc.getHistory().length}`);
}

main();
