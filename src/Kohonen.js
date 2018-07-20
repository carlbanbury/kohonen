import { scaleLinear } from 'd3-scale';
import { extent, mean, deviation } from 'd3-array';
import _ from 'lodash/fp';
import PCA from 'ml-pca';
import { dist, mult, diff, add } from './vector';
import n from 'norm.js';

// lodash/fp random has a fixed arity of 2, without the last (and useful) param
const random = _.random.convert({ fixed: false });

// lodash/fp map has an iteratee with a single arg
const mapWithIndex = _.map.convert({ cap: false });

// A basic implementation of Kohonen map

// The main class
class Kohonen {

  // The constructor needs two params :
  // * neurons : an already built neurons grid as an array
  // * data : data set to consider
  // * maxStep : the max step that will be clamped in scaleStepLearningCoef and
  //             scaleStepNeighborhood
  // * maxLearningCoef
  // * minLearningCoef
  // * maxNeighborhood
  // * minNeighborhood
  //
  // each neuron should provide a 2D vector pos,
  // which refer to the grid position
  //
  // You should use an hexagon grid as it is the easier case
  // to deal with neighborhood.
  //
  // You also should normalized your neighborhood in such a way that 2 neighbors
  // got an euclidian distance of 1 between each other.
  constructor({
    neurons,
    data,
    labels,
    maxStep = 10000,
    minLearningCoef = .1,
    maxLearningCoef = .4,
    minNeighborhood = .3,
    maxNeighborhood = 1,
    norm = true
  }) {

    // data vectors should have at least one dimension
    if (!data[0].length) {
      throw new Error('Kohonen constructor: data vectors should have at least one dimension');
    }

    // all vectors should have the same size
    // all vectors values should be number
    for (let ind in data) {
      if (data[ind].length !== data[0].length) {
        throw new Error('Kohonen constructor: all vectors should have the same size');
      }
      const allNum = _.reduce(
        (seed, current) => seed && !isNaN(current) && isFinite(current),
        true,
        data[ind]
      );
      if(!allNum) {
        throw new Error('Kohonen constructor: all vectors should number values');
      }
    }

    this.size = data[0].length;
    this.numNeurons = neurons.length;
    this.step = 0;
    this.maxStep = maxStep;
    this.norm = norm

    // generate scaleStepLearningCoef,
    // as the learning coef decreases with time
    this.scaleStepLearningCoef = scaleLinear()
      .clamp(true)
      .domain([0, maxStep])
      .range([maxLearningCoef, minLearningCoef]);

    // decrease neighborhood with time
    this.scaleStepNeighborhood = scaleLinear()
      .clamp(true)
      .domain([0, maxStep])
      .range([maxNeighborhood, minNeighborhood]);

    this._data = this.seedLabels(data, labels);

    // normalize data
    if (this.norm) {
      this._data.v = this.normalize(this._data.v);
    } 

    // On each neuron, generate a random vector v
    // of <size> dimension
    // each neuron has [{weight: <vector>, somdi: <vector>, pos: <vector>}]
    const randomInitialVectors = this.generateInitialVectors();
    this.neurons = neurons.map(function(neuron, index) {
      var item = randomInitialVectors[index];
      item.pos = neuron.pos;
      return item;
    });
  }

  // bind the labels and create SOMDI vectors
  seedLabels(data, labels) {
    var numClasses = _.max(labels) + 1;
    var out = {v: [], labels: [], somdi: []}
    data.map(function(item, index) {
      var somdi = [];

      if (labels) {
        var currentLabel = labels[index];
        var somdi = new Array(numClasses).fill(0);
        somdi[currentLabel] = 1;
      }

      out.v.push(item);
      out.labels.push(currentLabel);
      out.somdi.push(somdi);
    });

    return out;
  }

  normalize(data) {
    // TODO: Scale this properly between 0 and 1
    data.forEach(function(item, index) {
      data[index] = n.normalize(item, 'max');
    });
    return data;
  }

  // learn and return corresponding neurons for the dataset
  learn(log) {
    var self = this;
    for (var i=0; i<this.maxStep; i++) {
      // pick index for random sample
      var sampleIndex = this.pickDataIndex();
      var sample = this._data.v[sampleIndex];

      // find bmu
      const bmu = this.findBestMatchingUnit(sampleIndex);

      // compute current learning coef
      const currentLearningCoef = this.scaleStepLearningCoef(this.step);

      this.neurons.forEach(neuron => {
        // compute neighborhood
        const currentNeighborhood = self.neighborhood({ bmu, neuron });
        const scaleFactor = currentNeighborhood * currentLearningCoef;

        // update weights for neuron
        neuron.weight = self.updateStep(neuron.weight, sample, scaleFactor);

        // also update weights of SOMDI
        var sampleSOMDI = this._data.somdi[sampleIndex];
        neuron.somdi = self.updateStep(neuron.somdi, sampleSOMDI, scaleFactor);
      });

      log(this.neurons, this.step);
      this.step += 1;
    }
  }

  updateStep(weight, sample, scaleFactor) {
    // compute delta for the current neuron
    var error = diff(weight, sample);
    const delta = mult(error, scaleFactor);
    return add(weight, delta);
  }

  mapping() {
    return _.map(
      _.flow(
        this.findBestMatchingUnit.bind(this),
        _.get('pos'),
      ),
      this.data
    );
  }

  // expects an array of test samples and array of labels with corresponding indexes
  // e.g. testData = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]; testLabels = [1, 0, 2]
  predict(testData, testLabels) {
    var self;
    if (!this.labels) {
      return null;
    }

    // normalise the test data if norm enabled
    if (this.norm) {
      testData = self.normalize(testData)
    }

    var results = [];
    testData.forEach(function(item, index) {
       var bmu = self.findBestMatchingUnit(item);

       // SOMDI based calculation of winning neuron
       var somdi = bmu.somdi;
       var winningIndex = somdi.indexOf(_.max(somdi));
       results.push(winningIndex);
    });
  }

  weights() {
    return this.neurons;
  }

  // pick a random vector among data
  pickDataIndex() {
    return _.random(0, this._data.v.length - 1);
  }

  generateInitialVectors() {
    var output = [];
    for (var i=0; i<this.numNeurons; i++) {
      var vectorLength = this._data.v[0].length;

      var somdi = null;

      if (this.labels) {
        var somdiLength = this._data.somdi[0].length;
        var somdi = Array(somdiLength).fill(0).map(()=>Math.random());
      }
      
      output.push({
        weight: Array(vectorLength).fill(0).map(()=>Math.random()),
        somdi: somdi,
      });
    }

    return output;
  }

  // Find closer neuron
  findBestMatchingUnit(index) {
    var target = this._data.v[index];
    var _neurons = _.cloneDeep(this.neurons);

    return _.flow(
      _.orderBy(
        n => dist(target, n.weight),
        'asc',
      ),
      _.first
    )(this.neurons);
  }

  // http://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
  //
  // http://mathworld.wolfram.com/GaussianFunction.html
  //
  // neighborhood function made with a gaussian
  neighborhood({ bmu, n }) {
    const a = 1;
    const sigmaX = 1;
    const sigmaY = 1;

    return a
      * Math.exp(
        -(Math.pow(n.pos[0] - bmu.pos[0], 2) / 2 * Math.pow(sigmaX, 2) + Math.pow(n.pos[1] - bmu.pos[1], 2) / 2 * Math.pow(sigmaY, 2))
      )
      * this.scaleStepNeighborhood(this.step);
  }

  // The U-Matrix value of a particular node
  // is the average distance between the node's weight vector and that of its closest neighbors.
  // TODO: reimplement this
  // umatrix() {
  //   const roundToTwo = num => +(Math.round(num + "e+2") + "e-2");
  //   const findNeighors = cn => _.filter(
  //     n => roundToTwo(dist(n.pos, cn.pos)) === 1,
  //     this.neurons
  //   );
  //   return _.map(
  //     n => mean(findNeighors(n).map(nb => dist(nb.v, n.v))),
  //     this.neurons
  //   );
  // }


  // TODO: reimplement this
  // hitMap() {
  //   var that = this;
  //   var classMap = [];
  //   var positions = [];
  //   this.hitCount = [];
  //   var that = this;

  //   // loop through all data and match classes to positions
  //   this.data.forEach(function(item) {
  //     var classLabels = item.slice(-that.classPlanes.length);
  //     var bmu = that.findBestMatchingUnit(item);

  //     // store best matching unit and class index
  //     classMap.push([bmu.pos, classLabels]);
  //     positions.push(bmu.pos);
  //   });
    
  //   // loop through all positions
  //   positions.forEach(function(position) {
  //     // filter and sum class indexes to get hit count
  //     var matches = classMap.filter(function(result) {
  //       return result[0] === position;
  //     });

  //     var hits = new Array(that.classPlanes.length).fill(0);
  //     matches.forEach(function(match) {
  //       hits = add(hits, match[1]);
  //     });

  //     var winner = -1;
  //     var maxCount = _.max(hits);
  //     var guess = hits.indexOf(maxCount);
  //     var check = hits.lastIndexOf(maxCount);

  //     if (guess === check) {
  //       winner = guess;
  //     }

  //     var meta = {hits: hits, winner: winner};
  //     that.hitCount.push([position, meta]);
  //   });
  // }

  // TODO: reimplement this
  // classifyHits(test) {
  //   if (this.norm) {
  //       // make sure we only normalize the data and not the class planes!!!
  //       var classData = test.slice(-this.classPlanes.length);
  //       var testData = test.slice(0,test.length-this.classPlanes.length);

  //       testData = n.normalize(testData, 'max');
  //       test = testData.concat(classData);
  //   }
    

  //   if (!this.hitCount) {
  //     return null;
  //   }

  //   var bmu = this.findBestMatchingUnit(test);
  //   var match = this.hitCount.filter(function(item) {
  //     return item[0] === bmu.pos;
  //   });

  //   if (match) {
  //     return match[0];
  //   }

  //   return null;
  // }
}

export default Kohonen;
