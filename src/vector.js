import _ from 'lodash/fp';
import { randomNormal } from 'd3-random';
const math = require('mathjs');

// euclidian distance of 2 vectors
export const dist = (v1, v2) => {
  const d = Math.sqrt(v1.reduce((seed, cur, ind) => seed + Math.pow(v2[ind] - cur, 2), 0));
  if(isNaN(d) || !isFinite(d)){
    throw new Error('vector.dist : not a number');
  }
  return d;
};

export const dotProduct = function(v1, v2) {
  var sum = 0;
  v1.map(function(item, index) {
    sum += item * v2[index];
  });

  return sum;
}

// scalar mult of a vector
export const mult = (v, coef) => v.map(val => val * coef);

// scaler division
export const divide = (v, coef) => v.map(val => val / coef);

// scalar diff of a vector
export const diff = (v1, v2) => v1.map((val, i) => v2[i] - val);

// scalar addition of a vector
export const add = (v1, v2) => v1.map((val, i) => v2[i] + val);

// scale vector between 0 and 1
export const normalize = function(v, type) {
  if (type === 'zscore') {
    var mean = math.mean(v);
    var sigma = math.std(v);

    // standard scaling
    return v.map(function(x) {
      if (sigma === 0) {
        return 0;
      }
      return (x - mean) / sigma;
    });
  }

  var min = _.min(v);
  var max = _.max(v);

  var range = (max-min);

  return v.map(function(x) {
    if (range === 0) {
      return 0;
    }
    return (x - min) / range;
  });
};

// For a given size, return an array of `size` with random values
// within the gaussian normalization
//
// https://github.com/mbostock/d3/wiki/Math
export const random = (size, mean = 0.0, deviation = 1.0) =>
  _.map(i => randomNormal(mean, deviation)(), _.range(0, size));
