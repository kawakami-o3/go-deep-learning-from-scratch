package dataset

import (
	"fmt"
	"log"
	"testing"
)

func TestHelloWorld(t *testing.T) {
	// t.Fatal("not implemented")

	mnist, err := LoadMnist(true, true, false)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(len(mnist.TrainImg))
	fmt.Println(len(mnist.TrainImg[0]))
}
