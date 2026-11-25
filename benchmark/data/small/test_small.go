// Small Go file for benchmarking
package main

import "fmt"

func add(a int, b int) int {
    return a + b
}

func subtract(a int, b int) int {
    return a - b
}

type SimpleClass struct {
    value int
}

func (c *SimpleClass) increment() {
    c.value += 1
}

func (c *SimpleClass) getValue() int {
    return c.value
}

func main() {
    obj := &SimpleClass{value: 10}
    obj.increment()
    result := add(obj.getValue(), 5)
    fmt.Println(result)
}
