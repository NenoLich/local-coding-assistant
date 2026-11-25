// Medium Go file for benchmarking with more complex structure
package main

import (
	"errors"
	"fmt"
)

type OperationType string

const (
	Add      OperationType = "add"
	Subtract OperationType = "subtract"
	Multiply OperationType = "multiply"
	Divide   OperationType = "divide"
)

type OperationResult struct {
	Value         float64
	Operation     OperationType
	Success       bool
	ErrorMessage  string
}

type Calculator struct {
	history       []OperationResult
	currentValue  float64
}

func NewCalculator() *Calculator {
	return &Calculator{
		history:      make([]OperationResult, 0),
		currentValue: 0.0,
	}
}

func (c *Calculator) Add(x float64) OperationResult {
	c.currentValue += x
	result := OperationResult{
		Value:     c.currentValue,
		Operation: Add,
		Success:   true,
	}
	c.history = append(c.history, result)
	return result
}

func (c *Calculator) Subtract(x float64) OperationResult {
	c.currentValue -= x
	result := OperationResult{
		Value:     c.currentValue,
		Operation: Subtract,
		Success:   true,
	}
	c.history = append(c.history, result)
	return result
}

func (c *Calculator) Multiply(x float64) OperationResult {
	c.currentValue *= x
	result := OperationResult{
		Value:     c.currentValue,
		Operation: Multiply,
		Success:   true,
	}
	c.history = append(c.history, result)
	return result
}

func (c *Calculator) Divide(x float64) OperationResult {
	var result OperationResult
	if x == 0 {
		result = OperationResult{
			Value:         0,
			Operation:     Divide,
			Success:       false,
			ErrorMessage:  "Cannot divide by zero",
		}
	} else {
		c.currentValue /= x
		result = OperationResult{
			Value:     c.currentValue,
			Operation: Divide,
			Success:   true,
		}
	}
	c.history = append(c.history, result)
	return result
}

func (c *Calculator) GetHistory() []OperationResult {
	historyCopy := make([]OperationResult, len(c.history))
	copy(historyCopy, c.history)
	return historyCopy
}

func (c *Calculator) ClearHistory() {
	c.history = make([]OperationResult, 0)
}

func (c *Calculator) Reset() {
	c.currentValue = 0
	c.ClearHistory()
}

type ScientificCalculator struct {
	*Calculator
}

func NewScientificCalculator() *ScientificCalculator {
	return &ScientificCalculator{
		Calculator: NewCalculator(),
	}
}

func (sc *ScientificCalculator) Power(exponent float64) OperationResult {
	sc.currentValue = pow(sc.currentValue, exponent)
	result := OperationResult{
		Value:     sc.currentValue,
		Operation: Multiply,
		Success:   true,
	}
	sc.history = append(sc.history, result)
	return result
}

func (sc *ScientificCalculator) SquareRoot() OperationResult {
	var result OperationResult
	if sc.currentValue < 0 {
		result = OperationResult{
			Value:         0,
			Operation:     Multiply,
			Success:       false,
			ErrorMessage:  "Cannot calculate square root of negative number",
		}
	} else {
		sc.currentValue = sqrt(sc.currentValue)
		result = OperationResult{
			Value:     sc.currentValue,
			Operation: Multiply,
			Success:   true,
		}
	}
	sc.history = append(sc.history, result)
	return result
}

func performCalculationSequence(calc *Calculator) float64 {
	calc.Add(100)
	calc.Subtract(25)
	calc.Multiply(2)
	calc.Divide(3)
	return calc.currentValue
}

func pow(base, exponent float64) float64 {
	result := 1.0
	for i := 0; i < int(exponent); i++ {
		result *= base
	}
	return result
}

func sqrt(x float64) float64 {
	if x == 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func main() {
	calc := NewScientificCalculator()
	
	// Perform basic operations
	calc.Add(50)
	calc.Subtract(10)
	calc.Multiply(3)
	calc.Divide(4)
	
	// Perform scientific operations
	calc.Power(2)
	calc.SquareRoot()
	
	// Perform calculation sequence
	result := performCalculationSequence(calc.Calculator)
	
	fmt.Printf("Final result: %f\n", result)
	fmt.Printf("History length: %d\n", len(calc.GetHistory()))
}
