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
//
//
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
    maxStep = 10000,
    minLearningCoef = .1,
    maxLearningCoef = .4,
    minNeighborhood = .3,
    maxNeighborhood = 1,
    randomStart = false,
    classPlanes = undefined,
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
    this.randomStart = randomStart;
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

    this.classPlanes = classPlanes;
    this._data = data;

    // build structures for data including class planes and data without class planes
    if (this.classPlanes) {
      this.classData = this._data.map((item)=>{return item.slice(-this.classPlanes.length)});
      this._data = this._data.map((item)=>{return item.slice(0,item.length-this.classPlanes.length)});
    }

    // build normalized data
    if (this.norm) {
      this.data = this.normalize(this._data);
    } else {
      this.data = this._data;
    }
    

    // Append class information back to data now data has been normalized
    if (this.classPlanes) {
      for (var i=0; i<this.data.length; i++) {
        this.data[i] = this.data[i].concat(this.classData[i]);
      }
    }

    // then we store means and deviations for normalized datas
    this.means = _.flow(
      _.unzip,
      _.map(mean)
    )(this.data);

    this.deviations = _.flow(
      _.unzip,
      _.map(deviation)
    )(this.data);

    // On each neuron, generate a random vector v
    // of <size> dimension
    const randomInitialVectors = this.generateInitialVectors();
    this.neurons = mapWithIndex(
      (neuron, i) => ({
        ...neuron,
        v: randomInitialVectors[i],
      }),
      neurons
    );
  }

  normalize(data) {
    data.forEach(function(item, index) {
      data[index] = n.normalize(item, 'max');
    });
    return data;
  }

  // learn and return corresponding neurons for the dataset
  training(log = () => {
  }) {
    for (let i = 0; i < this.maxStep; i++) {
      // generate a random vector
      this.learn(this.generateLearningVector());
      log(this.neurons, this.step);
    }
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

  hitMap() {
    var that = this;
    var classMap = [];
    var positions = [];
    this.hitCount = [];
    var that = this;

    // loop through all data and match classes to positions
    this.data.forEach(function(item) {
      var classLabels = item.slice(-that.classPlanes.length);
      var bmu = that.findBestMatchingUnit(item);

      // store best matching unit and class index
      classMap.push([bmu.pos, classLabels]);
      positions.push(bmu.pos);
    });
    
    // loop through all positions
    positions.forEach(function(position) {
      // filter and sum class indexes to get hit count
      var matches = classMap.filter(function(result) {
        return result[0] === position;
      });

      var hits = new Array(that.classPlanes.length).fill(0);
      matches.forEach(function(match) {
        hits = add(hits, match[1]);
      });

      var meta = {hits: hits, winner: _.indexOf(hits, _.max(hits))};
      that.hitCount.push([position, meta]);
    });
  }

  classifyHits(test) {
    if (this.norm) {
        // make sure we only normalize the data and not the class planes!!!
        var classData = test.slice(-this.classPlanes.length);
        var testData = test.slice(0,test.length-this.classPlanes.length);

        testData = n.normalize(testData, 'max');
        test = testData.concat(classData);
    }
    

    if (!this.hitCount) {
      return null;
    }

    var bmu = this.findBestMatchingUnit(test);
    var match = this.hitCount.filter(function(item) {
      return item[0] === bmu.pos;
    });

    if (match) {
      return match[0];
    }

    return null;
  }

  classify(test, threshold) {
    if (!this.classPlanes) {
      return null;
    }

    if (this.norm) {
        // make sure we only normalize the data and not the class planes!!!
        var classData = test.slice(-this.classPlanes.length);
        var testData = test.slice(0,test.length-this.classPlanes.length);

        testData = n.normalize(testData, 'max');
        test = testData.concat(classData);
    }

    if (!threshold) {
      threshold = 0;
    }

    var bmu = this.findBestMatchingUnit(test);

    var classes = bmu.v.slice(bmu.v.length-this.classPlanes.length);
    var index = undefined;
    var temp = null;
    for (var i=0; i<classes.length; i++) {
      if (classes[i] > temp && classes[i] > threshold) {
        temp = classes[i];
        index = i;
      }
    }

    return {className: this.classPlanes[index], index: index, weight: temp};
  }

  weights() {
    return this.neurons;
  }

  // The U-Matrix value of a particular node
  // is the average distance between the node's weight vector and that of its closest neighbors.
  umatrix() {
    const roundToTwo = num => +(Math.round(num + "e+2") + "e-2");
    const findNeighors = cn => _.filter(
      n => roundToTwo(dist(n.pos, cn.pos)) === 1,
      this.neurons
    );
    return _.map(
      n => mean(findNeighors(n).map(nb => dist(nb.v, n.v))),
      this.neurons
    );
  }

  // pick a random vector among data
  generateLearningVector() {
    return this.data[_.random(0, this.data.length - 1)];
  }

  generateInitialVectors() {

    // use random initialisation instead of PCA
    if (this.randomStart) {
      var output = [];
      for (var i=0; i<this.numNeurons; i++) {
        var tempVector = Array(this.data[0].length).fill(0).map(()=>Math.random());
        output.push(tempVector);
      }

      return output;
    }

    // principal component analysis
    // standardize to false as we already standardize ours
    //
    const pca = new PCA(this.data, {
      center: true,
      scale: false,
    });

    // we'll only keep the 2 largest eigenvectors
    const transposedEV = _.take(2, pca.getLoadings());

    // function to generate random vectors into eigenvectors space
    const generateRandomVecWithinEigenvectorsSpace = () => add(
      mult(transposedEV[0], random(-.5, .5, true)),
      mult(transposedEV[1], random(-.5, .5, true))
    );

    // we generate all random vectors and uncentered them by adding means vector
    return _.map(
      () => add(generateRandomVecWithinEigenvectorsSpace(), this.means),
      _.range(0, this.numNeurons)
    );
  }

  learn(v) {
    // find bmu
    const bmu = this.findBestMatchingUnit(v);
    // compute current learning coef
    const currentLearningCoef = this.scaleStepLearningCoef(this.step);

    this.neurons.forEach(n => {
      // compute neighborhood
      const currentNeighborhood = this.neighborhood({ bmu, n });

      // compute delta for the current neuron
      const delta = mult(
        diff(n.v, v),
        currentNeighborhood * currentLearningCoef
      );

      // update current vector
      n.v = add(n.v, delta);
    });
    this.step += 1;
  }

  // Find closer neuron
  findBestMatchingUnit(v) {
    var target = v;
    var _neurons = _.cloneDeep(this.neurons);

    if (this.classPlanes) {
      // do not include class plane data in finding best matching unit.
      target = target.slice(0, target.length-this.classPlanes.length);

      _neurons.map((item) => {
        item.v = item.v.slice(0, item.v.length-this.classPlanes.length);
      });
    }

    var bmuTruncated = _.flow(
      _.orderBy(
        n => dist(target, n.v),
        'asc',
      ),
      _.first
    )(_neurons)

    if (!this.classPlanes) {
      return bmuTruncated;
    }

    var output = null;
    this.neurons.forEach(function(item) {
      if (item.pos[0] === bmuTruncated.pos[0] && item.pos[1] === bmuTruncated.pos[1]) {
        output = item;
      }
    });

    return output;
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

}

export default Kohonen;
