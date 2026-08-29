// Test class for JavaScript AST parsing
class TestClass {
  constructor(name) {
    this.name = name;
  }

  // Biased method called by many others
  biasedMethod() {
    console.log('Biased method called');
  }

  // Unbiased methods that call biasedMethod
  unbiasedMethod1() {
    this.biasedMethod();
  }

  unbiasedMethod2() {
    this.biasedMethod();
  }

  unbiasedMethod3() {
    this.biasedMethod();
  }

  unbiasedMethod4(param) {
    this.biasedMethod();
    console.log(param);
  }

  unbiasedMethod5() {
    for (let i = 0; i < 3; i++) {
      this.biasedMethod();
    }
  }

  // Method with arrow function
  methodWithArrow() {
    const arrow = () => {
      this.biasedMethod();
    };
    arrow();
  }

  // Async method
  async asyncMethod() {
    await this.biasedMethod();
  }

  // Method with callback
  methodWithCallback(callback) {
    callback();
    this.biasedMethod();
  }
}

// Outsider class
class OutsiderClass {
  callTestClass1() {
    const test = new TestClass('test');
    test.biasedMethod();
  }

  callTestClass2() {
    const test = new TestClass('test');
    test.biasedMethod();
  }

  callTestClass3() {
    const test = new TestClass('test');
    test.biasedMethod();
  }
}

// Regular function
function regularFunction() {
  console.log('Regular function');
}

// Function with parameters
function functionWithParams(a, b) {
  return a + b;
}

// Arrow function
const arrowFunction = () => {
  console.log('Arrow function');
};

// Arrow function with parameters
const arrowWithParams = (a, b) => a + b;

// Higher-order function
function higherOrderFunction(fn) {
  fn();
}

// Callback example
function callbackExample(callback) {
  callback();
}

// Promise example
function promiseExample() {
  return new Promise((resolve, reject) => {
    resolve('Success');
  });
}

// Async function
async function asyncFunction() {
  const result = await promiseExample();
  console.log(result);
}

// Object with methods
const objectWithMethods = {
  method1() {
    console.log('Method 1');
  },
  method2() {
    this.method1();
  },
};

// Class inheritance
class BaseClass {
  baseMethod() {
    console.log('Base method');
  }
}

class DerivedClass extends BaseClass {
  derivedMethod() {
    this.baseMethod();
  }
}

// Static method
class ClassWithStatic {
  static staticMethod() {
    console.log('Static method');
  }
}

// Getter and setter
class ClassWithGetterSetter {
  constructor() {
    this._value = 0;
  }

  get value() {
    return this._value;
  }

  set value(val) {
    this._value = val;
  }
}

// Class with private fields
class ClassWithPrivate {
  #privateField = 'private';

  getPrivate() {
    return this.#privateField;
  }
}

// Destructuring
function destructuringExample({ name, age }) {
  console.log(name, age);
}

// Spread operator
function spreadExample(...args) {
  console.log(args);
}

// Template literal
function templateLiteralExample(name) {
  return `Hello ${name}`;
}

// Default parameters
function defaultParams(a = 1, b = 2) {
  return a + b;
}

// Rest parameters
function restParams(...args) {
  return args.reduce((sum, val) => sum + val, 0);
}

// Immediately invoked function expression
(function() {
  console.log('IIFE');
})();

// Module pattern
const modulePattern = (function() {
  let privateVar = 0;

  return {
    increment() {
      privateVar++;
    },
    getCount() {
      return privateVar;
    },
  };
})();

// Prototype
function PrototypeClass() {
  this.value = 0;
}

PrototypeClass.prototype.increment = function() {
  this.value++;
};

// Class with computed property names
const methodName = 'computed';
class ClassWithComputed {
  [methodName]() {
    console.log('Computed method');
  }
}

// Generator function
function* generatorFunction() {
  yield 1;
  yield 2;
  yield 3;
}

// Main execution
const test = new TestClass('main');
test.biasedMethod();
test.unbiasedMethod1();
test.unbiasedMethod2();
test.unbiasedMethod3();
