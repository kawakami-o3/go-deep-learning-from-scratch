package main

import (
	"fmt"

	"gorgonia.org/tensor"
)

type MulLayerFloat struct {
	x float64
	y float64
}

func (this *MulLayerFloat) forward(x, y float64) float64 {
	this.x = x
	this.y = y
	out := x * y

	return out
}

func (this *MulLayerFloat) backward(dout float64) (float64, float64) {
	dx := dout * this.y
	dy := dout * this.x

	return dx, dy
}

func runMulLayerFloat() {
	apple := 100.0
	appleNum := 2.0
	tax := 1.1

	mulAppleLayer := &MulLayerFloat{}
	mulTaxLayer := &MulLayerFloat{}

	applePrice := mulAppleLayer.forward(apple, appleNum)
	price := mulTaxLayer.forward(applePrice, tax)

	fmt.Println(price)

	dprice := 1.0
	dapplePrice, dtax := mulTaxLayer.backward(dprice)
	dapple, dappleNum := mulAppleLayer.backward(dapplePrice)

	fmt.Println(dapple, dappleNum, dtax)
}

type AddLayerFloat struct {
}

func (this *AddLayerFloat) forward(x, y float64) float64 {
	out := x + y
	return out
}

func (this *AddLayerFloat) backward(dout float64) (float64, float64) {
	dx := dout * 1.0
	dy := dout * 1.0
	return dx, dy
}

func runLayers() {
	apple := 100.0
	appleNum := 2.0
	orange := 150.0
	orangeNum := 3.0
	tax := 1.1

	// layer
	mulAppleLayer := &MulLayerFloat{}
	mulOrangeLayer := &MulLayerFloat{}
	addAppleOrangeLayer := &AddLayerFloat{}
	mulTaxLayer := &MulLayerFloat{}

	// forward
	applePrice := mulAppleLayer.forward(apple, appleNum)
	orangePrice := mulOrangeLayer.forward(orange, orangeNum)
	allPrice := addAppleOrangeLayer.forward(applePrice, orangePrice)
	price := mulTaxLayer.forward(allPrice, tax)

	// backward
	dprice := 1.0
	dallPrice, dtax := mulTaxLayer.backward(dprice)
	dapplePrice, dorangePrice := addAppleOrangeLayer.backward(dallPrice)
	dorange, dorangeNum := mulOrangeLayer.backward(dorangePrice)
	dapple, dappleNum := mulAppleLayer.backward(dapplePrice)

	fmt.Println(price)
	fmt.Println(dapple, dappleNum, dorange, dorangeNum, dtax)
}

type Relu struct {
	mask []int
}

func (this *Relu) forward(x *tensor.Dense) *tensor.Dense {
	it := x.Iterator()
	fmt.Println(it)
	for !it.Done() {
		idx, _ := it.Next()
		a := x.Get(idx).(float64)
		if a < 0 {
			x.Set(idx, 0.0)
			this.mask = append(this.mask, idx)
		}
	}
	return x
}

func (this *Relu) backward(dout *tensor.Dense) *tensor.Dense {
	for _, idx := range this.mask {
		dout.Set(idx, 0.0)
	}
	return dout
}

func runRelu() {
	x := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, -0.5, -2.0, 3.0}))

	relu := &Relu{}

	relu.forward(x)
	fmt.Println(x)
}

func main() {
	//runMulLayerFloat()
	runRelu()
	//runLayers()
}
