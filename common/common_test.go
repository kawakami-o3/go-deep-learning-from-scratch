package common

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func TestAddVector(t *testing.T) {

	//m := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 6)))
	m := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 10, 10, 10}))
	fmt.Println(m)

	v := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float64{1, 2, 3}))
	fmt.Println(v)

	AddVector(m, v)

	fmt.Println()
	fmt.Println(m)
	fmt.Println(v)
}
