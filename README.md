# kohonen
A basic implementation of a Kohonen map in JavaScript, forked from here:
https://github.com/seracio/kohonen

Beyond the basic Kohonen map or self organising map (SOM), the SOM discriminant index [SOMDI](https://www.researchgate.net/publication/223686662_Self_Organising_Maps_for_variable_selection_Application_to_human_saliva_analysed_by_nuclear_magnetic_resonance_spectroscopy_to_investigate_the_effect_of_an_oral_healthcare_product) is implemented to provide variable selection.

This has further been extended to provide classification based on SOMDI to identify the winning class for each neuron, and learning vector quantization (LVQ) as a method of supervised learning that can be layered in.

## Example

Continue below for information on how to use the Kohonen class, or see the [Raman Tools](https://github.com/cbanbury/raman-tools) reposistory for a web app GUI using these functions. An example of the application can be found at:

http://raman.banbury.ch

## Usage

### Import lib

```
npm i https://github.com/cbanbury/kohonen --save
```

Then, in your JS script :

```javascript
import Kohonen, {hexagonHelper} from 'kohonen';
```

### API

#### Kohonen

The Kohonen class is the main class.

##### Constructor

|  param name      | definition       | type             | mandatory        | default          | options |
|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:--------|
|    neurons       |  grid of neurons |   Array          |       yes        |                  |
|    data          |  dataset         |   Array of Array |       yes        |                  |
|    labels        |  datset          |   Array          |       no         |                  |
|    maxStep       | step max to clamp|   Number         |       no         |     10000        |
| maxLearningCoef  |                  |   Number         |       no         |      1           |
| minLearningCoef  |                  |   Number         |       no         |      .3          |
| maxNeighborhood  |                  |   Number         |       no         |      1           |
| minNeighborhood  |                  |   Number         |       no         |      .3          |
|    norm          |normalisation type|   String         |       no         |                  | 'zcore', 'max'
|    distance      |  distance metric |   String         |       no         |     Euclidian    | 'manhattan'
|   class_method   | classificatin method |   String         |       no     |     somdi        | 'hits'

```javascript

// instanciate your Kohonen map
const k = new Kohonen({data, neurons});

// you can use the grid helper to generate a grid with 10x10 hexagons
const k = new Kohonen({data, neurons: hexagonHelper.generateGrid(10,10)});
```

`neurons` parameter should be a flat array of `{ pos: [x,y] }`. `pos` array being the coordinate on the grid.

`data` parameter is an array of the vectors you want to display. There is no need to standardize your data, that will
 be done internally by scaling each feature to the [0,1] range.

 `labels` parameter is an array of integer labels to describe the class that each sample in the data belongs to.

The function of the constructor is:

* standardize the given data set
* initialize random weights for neurons
* bind data and labels, create additional weights for SOMDI if applicable

##### training method

|  param name      | definition                                       | type             | mandatory        | default          |
|:----------------:|:------------------------------------------------:|:----------------:|:----------------:|:----------------:|
|    log           |  func called after each step of learning process |   Function       |       no         |  ()=>{}          |


```javascript
k.training();
```

`training` method iterates on random vectors picked on normalized data.
If a log function is provided as a parameter, it will receive instance neurons and step as params.

##### mapping method

`mapping` method returns grid position for each data provided on the constructor.

```javascript
const myPositions = k.mapping();
```

##### predict method
`predict` method returns predictions for some test data, using the SOM

```javascript
// train the model
k.training();

// make some predictions
var testData = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.9, 0, 0];
var testLabels = [0, 1, 2, 0];

var predictions = k.predict(testData, testlabels);
```
##### SOMDI method
`SOMDI` method returns the SOMDI for a given class label

```javascript
// train the model
k.training();

// compute SOMDI for given class label
var somdi = k.SOMDI(0);
```
