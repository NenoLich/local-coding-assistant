package main

import "fmt"

// TestStruct represents a test structure
type TestStruct struct {
	Name string
}

// NewTestStruct creates a new TestStruct
func NewTestStruct(name string) *TestStruct {
	return &TestStruct{Name: name}
}

// BiasedMethod is called by many other methods
func (t *TestStruct) BiasedMethod() {
	fmt.Println("Biased method called")
}

// UnbiasedMethod1 calls BiasedMethod
func (t *TestStruct) UnbiasedMethod1() {
	t.BiasedMethod()
}

// UnbiasedMethod2 calls BiasedMethod
func (t *TestStruct) UnbiasedMethod2() {
	t.BiasedMethod()
}

// UnbiasedMethod3 calls BiasedMethod
func (t *TestStruct) UnbiasedMethod3() {
	t.BiasedMethod()
}

// UnbiasedMethod4 calls BiasedMethod with parameters
func (t *TestStruct) UnbiasedMethod4(param string) {
	t.BiasedMethod()
	fmt.Println(param)
}

// UnbiasedMethod5 calls BiasedMethod in a loop
func (t *TestStruct) UnbiasedMethod5() {
	for i := 0; i < 3; i++ {
		t.BiasedMethod()
	}
}

// OutsiderStruct represents an outsider structure
type OutsiderStruct struct{}

// CallTestStruct1 calls TestStruct's BiasedMethod
func (o *OutsiderStruct) CallTestStruct1() {
	test := NewTestStruct("test")
	test.BiasedMethod()
}

// CallTestStruct2 calls TestStruct's BiasedMethod
func (o *OutsiderStruct) CallTestStruct2() {
	test := NewTestStruct("test")
	test.BiasedMethod()
}

// CallTestStruct3 calls TestStruct's BiasedMethod
func (o *OutsiderStruct) CallTestStruct3() {
	test := NewTestStruct("test")
	test.BiasedMethod()
}

// InterfaceExample demonstrates interface usage
type ExampleInterface interface {
	InterfaceMethod()
}

// InterfaceImpl implements ExampleInterface
type InterfaceImpl struct{}

func (i *InterfaceImpl) InterfaceMethod() {
	fmt.Println("Interface method")
}

func CallInterfaceMethod(e ExampleInterface) {
	e.InterfaceMethod()
}

// Function with multiple return values
func MultiReturn() (int, error) {
	return 42, nil
}

// Function with variadic parameters
func VariadicFunc(args ...string) {
	for _, arg := range args {
		fmt.Println(arg)
	}
}

// Anonymous function
func AnonymousFuncExample() {
	func() {
		fmt.Println("Anonymous function")
	}()
}

// Closure example
func ClosureExample() func() int {
	count := 0
	return func() int {
		count++
		return count
	}
}

// Deferred function
func DeferredExample() {
	defer func() {
		fmt.Println("Deferred")
	}()
	fmt.Println("Main")
}

// Goroutine example
func GoroutineExample() {
	go func() {
		fmt.Println("Goroutine")
	}()
}

// Channel example
func ChannelExample() {
	ch := make(chan int)
	go func() {
		ch <- 42
	}()
	<-ch
}

// Struct with embedded type
type EmbeddedStruct struct {
	*TestStruct
}

func (e *EmbeddedStruct) CallEmbedded() {
	e.BiasedMethod()
}

// Method with pointer receiver
func (t *TestStruct) PointerMethod() {
	fmt.Println(t.Name)
}

// Method with value receiver
func (t TestStruct) ValueMethod() {
	fmt.Println(t.Name)
}

// Generic function (Go 1.18+)
func GenericFunc[T any](value T) T {
	return value
}

// Type alias
type MyInt = int

// Constant
const MyConst = 100

// Variable
var MyVar string = "test"

// Init function
func init() {
	fmt.Println("Initialized")
}

// Main function
func main() {
	test := NewTestStruct("main")
	test.BiasedMethod()
	test.UnbiasedMethod1()
	test.UnbiasedMethod2()
	test.UnbiasedMethod3()
}
