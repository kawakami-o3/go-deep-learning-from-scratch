package dataset

import (
	"fmt"
	"log"
	"testing"
)

func TestHelloWorld(t *testing.T) {

	mnist, err := LoadMnist(false, false, false)
	if err != nil {
		log.Fatal(err)
	}

	for _, number := range mnist.TrainImg2d {
		for _, bytes := range number {
			for _, b := range bytes {
				if b == 0 {
					fmt.Print("  ")
				} else {
					fmt.Print(" *")
				}
			}
			fmt.Println("|")
		}
		break
		fmt.Println()
	}
}
