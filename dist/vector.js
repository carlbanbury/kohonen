'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.normalize = exports.add = exports.diff = exports.divide = exports.mult = exports.dist = undefined;

var _fp = require('lodash/fp');

var _fp2 = _interopRequireDefault(_fp);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var math = require('mathjs');

// euclidian distance of 2 vectors
var dist = exports.dist = function dist(v1, v2) {
  var d = Math.sqrt(v1.reduce(function (seed, cur, ind) {
    return seed + Math.pow(v2[ind] - cur, 2);
  }, 0));
  if (isNaN(d) || !isFinite(d)) {
    throw new Error('vector.dist : not a number');
  }
  return d;
};

// scalar mult of a vector
var mult = exports.mult = function mult(v, coef) {
  return v.map(function (val) {
    return val * coef;
  });
};

// scaler division
var divide = exports.divide = function divide(v, coef) {
  return v.map(function (val) {
    return val / coef;
  });
};

// scalar diff of a vector
var diff = exports.diff = function diff(v1, v2) {
  return v1.map(function (val, i) {
    return v2[i] - val;
  });
};

// scalar addition of a vector
var add = exports.add = function add(v1, v2) {
  return v1.map(function (val, i) {
    return v2[i] + val;
  });
};

// scale vector between 0 and 1
var normalize = exports.normalize = function normalize(v, type) {
  if (type === 'zscore') {
    var mean = math.mean(v);
    var sigma = math.std(v);

    // standard scaling
    return v.map(function (x) {
      if (sigma === 0) {
        return 0;
      }
      return (x - mean) / sigma;
    });
  }

  var min = _fp2.default.min(v);
  var max = _fp2.default.max(v);

  var range = max - min;

  return v.map(function (x) {
    if (range === 0) {
      return 0;
    }
    return (x - min) / range;
  });
};