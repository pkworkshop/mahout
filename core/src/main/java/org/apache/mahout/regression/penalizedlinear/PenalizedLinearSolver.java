/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.regression.penalizedlinear;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PenalizedLinearSolver {

  private static final Logger log = LoggerFactory.getLogger(PenalizedLinearSolver.class);
  private double[] lambda;
  private String lambdaString;
  private int dimension;
  private boolean intercept;
  private double alpha;
  private Coefficients[] coefficients;
  private TrainTestData[] trainTestData;
  private TrainTestData wholeData;
  private double[] optLambda;
  double[] trainError;
  double[] testError;

  public double[] getLambda() {
    return lambda;
  }

  public double[] getTrainError() {
    return trainError;
  }

  public double[] getTestError() {
    return testError;
  }

  public Coefficients[] getCoefficients() {
    return coefficients;
  }

  public double[] getOptLambda() {
    return optLambda;
  }

  public void setIntercept(boolean intercept) {
    this.intercept = intercept;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  public void setLambdaString(String lambdaString) {
    this.lambdaString = lambdaString;
  }

  private void setLambda() {
    String[] lambdaSeq = lambdaString.split(",");
    lambda = new double[lambdaSeq.length];
    for(int i = 0;i < lambda.length; ++i) {
      lambda[i] = Double.valueOf(lambdaSeq[i]);
    }
  }

  private void setDefaultLambda() {
    lambda = new double[PenalizedLinearConstants.DEFAULT_NUM_LAMBDA];
    double[] correlation = new double[dimension];
    double ratio = 0;
    ratio = Math.max(alpha, PenalizedLinearConstants.ALPHA_THRESHOLD);
    if(intercept) {
      for(int i = 0;i < dimension; ++i) {
        correlation[i] = Math.abs(wholeData.trainXY[i] - wholeData.trainX[i] * wholeData.trainY);
      }
      for(int i = 0;i < dimension; ++i) {
        if(wholeData.trainXcXc[i][i] < PenalizedLinearConstants.MINIMUM_VARIANCE) {
          correlation[i] = 0;
        }
        else {
          correlation[i] /= (Math.sqrt(wholeData.trainXcXc[i][i]) * ratio);
        }
      }
    }
    else {
      for(int i = 0;i < dimension; ++i) {
        correlation[i] = Math.abs(wholeData.trainXY[i]);
      }
      for(int i = 0;i < dimension; ++i) {
        if(wholeData.trainXcXc[i][i] < PenalizedLinearConstants.MINIMUM_VARIANCE) {
          correlation[i] = 0;
        }
        else {
          correlation[i] /= (Math.sqrt(wholeData.trainXcXc[i][i]) * ratio);
        }
      }
    }
    int index = 0;
    for(int i = 1;i < correlation.length; ++i) {
      if(correlation[index] < correlation[i]) {
        index = i;
      }
    }
    double lambdaMin = correlation[index] * PenalizedLinearConstants.LAMBDA_RATIO * PenalizedLinearConstants.LAMBDA_MULTIPLIER;
    for(int i = 0;i < lambda.length; ++i) {
      lambda[i] = lambdaMin * Math.exp(Math.log(PenalizedLinearConstants.LAMBDA_RATIO) / (lambda.length - 1) * i * (-1));
    }
  }

  private double averageSquareError(double[] x, double[] y) {
    double error = 0.0;
    int length = x.length;
    for (int i = 0; i < length; ++i) {
      error += (x[i] - y[i]) * (x[i] - y[i]) / length;
    }
    return Math.sqrt(error);
  }

  public class Coefficients {
    public double beta0;
    public double[] beta = new double[dimension];
  }

  private class TrainTestError {
    TrainTestError() {
      trainError = 0.0;
      testError = 0.0;
    }
    public double trainError;
    public double testError;
  }

  private TrainTestError averageSumOfSquare(TrainTestData trainTestData, double[] beta, double beta0) {
    TrainTestError trainTestError = new TrainTestError();
    for(int i = 0;i < dimension; ++i) {
      for(int j = 0;j < dimension; ++j) {
        trainTestError.trainError += trainTestData.trainXcXc[i][j] * beta[i] * beta[j];
        trainTestError.trainError += trainTestData.trainX[i] * trainTestData.trainX[j] * beta[i] * beta[j];
      }
    }

    for(int i = 0;i < dimension; ++i) {
      trainTestError.trainError -= 2 * beta[i] * (trainTestData.trainXY[i] - beta0 * trainTestData.trainX[i]);
    }

    trainTestError.trainError += (trainTestData.trainYY - 2 * beta0 * trainTestData.trainY + beta0 * beta0);
    trainTestError.trainError = Math.max(0.0, trainTestError.trainError);

    if(trainTestData.testXcXc != null) {
      for(int i = 0;i < dimension; ++i) {
        for(int j = 0;j < dimension; ++j) {
          trainTestError.testError += trainTestData.testXcXc[i][j] * beta[i] * beta[j];
          trainTestError.testError += trainTestData.testX[i] * trainTestData.testX[j] * beta[i] * beta[j];
        }
      }

      for(int i = 0;i < dimension; ++i) {
        trainTestError.testError -= 2 * beta[i] * (trainTestData.testXY[i] - beta0 * trainTestData.testX[i]);
      }

      trainTestError.testError += (trainTestData.testYY - 2 * beta0 * trainTestData.testY + beta0 * beta0);
      trainTestError.testError = Math.max(0.0, trainTestError.testError);
    }
    return trainTestError;
  }

  /**
   * compute the regularized path of the coefficients with \lambda sequence.
   * @param lambda
   */
  public void regularizePath(double[] lambda) {
    regularizePath(wholeData, lambda, alpha);
  }

  private void regularizePath(TrainTestData trainTestData, double[] lambdas, double alpha) {
    boolean[] mask = new boolean[dimension];
    coefficients = new Coefficients[lambdas.length];
    for(int i = 0;i < coefficients.length; ++i) {
      coefficients[i] = new Coefficients();
    }
    if(intercept) {
      for (int i = 0; i < dimension; ++i) {
        if (trainTestData.trainXcXc[i][i] < PenalizedLinearConstants.MINIMUM_VARIANCE) {
          mask[i] = false;
        }
        else {
          mask[i] = true;
        }
      }
      double[] beta = new double[dimension];
      double[][] A = trainTestData.trainXcXc;
      double[] B = new double[dimension];
      for(int i = 0;i < dimension; ++i) {
        B[i] = trainTestData.trainXY[i] - trainTestData.trainX[i] * trainTestData.trainY;
      }
      double beta0 = 0.0;
      for(int i = 0;i < lambdas.length; ++i) {
        beta = coordinateDescentSolver(trainTestData, A, B, alpha, lambdas[i], beta, mask);
        beta0 = trainTestData.trainY;
        for(int j = 0;j < dimension; ++j) {
          beta0 -= trainTestData.trainX[j] * beta[j];
        }
        coefficients[i].beta = beta;
        coefficients[i].beta0 = beta0;
      }
    }
    else {
      double[] beta = new double[dimension];
      double[][] A = new double[dimension][dimension];
      double[] B = trainTestData.trainXY;
      for(int i = 0;i < dimension; ++i) {
        A[i][i] = trainTestData.trainXcXc[i][i] + trainTestData.trainX[i] * trainTestData.trainX[i];
        for(int j = i + 1;j < dimension; ++j) {
          A[i][j] = trainTestData.trainXcXc[i][j] + trainTestData.trainX[i] * trainTestData.trainX[j];
          A[j][i] = A[i][j];
        }
      }
      for(int i = 0;i < dimension; ++i) {
        if(A[i][i] < PenalizedLinearConstants.MINIMUM_VARIANCE) {
          mask[i] = false;
        }
        else {
          mask[i] = true;
        }
      }

      double beta0 = 0.0;
      for(int i = 0;i < lambdas.length; ++i) {
        beta = coordinateDescentSolver(trainTestData, A, B, alpha, lambdas[i], beta, mask);
        coefficients[i].beta = beta;
        coefficients[i].beta0 = beta0;
      }
    }
  }

  private TrainTestError[] linearSolver(TrainTestData trainTestData, double[] lambdas, double alpha) {
    TrainTestError[] trainTestError = new TrainTestError[lambdas.length];
    boolean[] mask = new boolean[dimension];
    if(intercept) {
      for (int i = 0; i < dimension; ++i) {
        if (trainTestData.trainXcXc[i][i] < PenalizedLinearConstants.MINIMUM_VARIANCE) {
          mask[i] = false;
        }
        else {
          mask[i] = true;
        }
      }
      double[] beta = new double[dimension];
      double[][] A = trainTestData.trainXcXc;
      double[] B = new double[dimension];
      for(int i = 0;i < dimension; ++i) {
        B[i] = trainTestData.trainXY[i] - trainTestData.trainX[i] * trainTestData.trainY;
      }

      double beta0 = 0.0;
      for(int i = 0;i < lambdas.length; ++i) {
        beta = coordinateDescentSolver(trainTestData, A, B, alpha, lambdas[i], beta, mask);
        beta0 = trainTestData.trainY;
        for(int j = 0;j < dimension; ++j) {
          beta0 -= trainTestData.trainX[j] * beta[j];
        }
        trainTestError[i] = averageSumOfSquare(trainTestData, beta, beta0);
      }
    }
    else {
      double[] beta = new double[dimension];
      double[][] A = new double[dimension][dimension];
      double[] B = trainTestData.trainXY;
      for(int i = 0;i < dimension; ++i) {
        A[i][i] = trainTestData.trainXcXc[i][i] + trainTestData.trainX[i] * trainTestData.trainX[i];
        for(int j = i + 1;j < dimension; ++j) {
          A[i][j] = trainTestData.trainXcXc[i][j] + trainTestData.trainX[i] * trainTestData.trainX[j];
          A[j][i] = A[i][j];
        }
      }
      for(int i = 0;i < dimension; ++i) {
        if(A[i][i] < PenalizedLinearConstants.MINIMUM_VARIANCE) {
          mask[i] = false;
        }
        else {
          mask[i] = true;
        }
      }

      double beta0 = 0.0;
      for(int i = 0;i < lambdas.length; ++i) {
        beta = coordinateDescentSolver(trainTestData, A, B, alpha, lambdas[i], beta, mask);
        trainTestError[i] = averageSumOfSquare(trainTestData, beta, beta0);
      }
    }
    return trainTestError;
  }

  /**
   * Solve the optimization
   * 1/2 * \beta' * A * \beta - \beta' * B + \lambda * ((1 - \alpha) / 2 * ||\beta||_2^2 + \alpha * ||\beta||_1)
   * by coordinate descent, where A is semi-positive definite.
   *
   * @param trainTestData the data structure to store the training and testing data
   * @param A
   * @param B
   * @param alpha
   * @param lambda
   * @param beta
   * @param mask
   * @return
   */
  private double[] coordinateDescentSolver(TrainTestData trainTestData, double[][] A, double[] B, double alpha, double lambda, double[] beta, boolean[] mask) {
    double[] betaUpdate = beta.clone();
    double[] betaPrevious = new double[betaUpdate.length];
    int p = betaUpdate.length;
    for (int i = 0; i < PenalizedLinearConstants.MAXIMUM_PASS; ++i) {
      System.arraycopy(betaUpdate, 0, betaPrevious, 0, betaUpdate.length);
      for (int j = 0; j < p; ++j) {
        if (mask[j]) {
          double z = B[j];
          for (int k = 0; k < p; ++k) {
            if (j != k && mask[k]) {
              z -= A[j][k] * betaUpdate[k];
            }
          }
          double r = lambda * alpha * Math.sqrt(trainTestData.trainXcXc[j][j]);
          betaUpdate[j] = Math.signum(z) * Math.max(Math.abs(z) - r, 0.0)
                  / (A[j][j] + trainTestData.trainXcXc[j][j] * lambda * (1.0 - alpha));
        }
      }
      if (averageSquareError(betaUpdate, betaPrevious) < PenalizedLinearConstants.CONVERGENCE_THRESHOLD) {
        break;
      }
    }
    return betaUpdate;
  }

  private class TrainTestData {
    public double[][] trainXcXc;
    public double[] trainXY;
    public double[] trainX;
    public double trainYY;
    public double trainY;
    public double trainCount;
    public double[][] testXcXc;
    public double[] testXY;
    public double[] testX;
    public double testYY;
    public double testY;
    public double testCount;

    TrainTestData() {
    }

    public void initTrain() {
      trainXcXc = new double[dimension][dimension];
      trainXY = new double[dimension];
      trainX = new double[dimension];
    }

    public void initTest() {
      testXcXc = new double[dimension][dimension];
      testXY = new double[dimension];
      testX = new double[dimension];
    }

    public void setTrain(Vector train) {
      int index = 0;
      for (int i = 0; i < dimension; ++i) {
        trainXY[i] = train.get(index++);
      }
      trainY = train.get(index++);
      for (int i = 0; i < dimension; ++i) {
        trainX[i] = train.get(index++);
      }
      for (int i = 0; i < dimension; ++i) {
        for (int j = i; j < dimension; ++j) {
          trainXcXc[i][j] = train.get(index++);
          trainXcXc[j][i] = trainXcXc[i][j];
        }
      }
      trainYY = train.get(index++);
      trainCount = train.get(index++);
    }

    public void setTest(Vector test) {
      int index = 0;
      for (int i = 0; i < dimension; ++i) {
        testXY[i] = test.get(index++);
      }
      testY = test.get(index++);
      for (int i = 0; i < dimension; ++i) {
        testX[i] = test.get(index++);
      }
      for (int i = 0; i < dimension; ++i) {
        for (int j = i; j < dimension; ++j) {
          testXcXc[i][j] = test.get(index++);
          testXcXc[j][i] = testXcXc[i][j];
        }
      }
      testYY = test.get(index++);
      testCount = test.get(index++);
    }
  }

  private TrainTestData fillTrainTest(TrainTestData[] trainTestData, List<Vector> dataTable) {
    for(int i = 0;i < trainTestData.length; ++i) {
      trainTestData[i] = new TrainTestData();
    }

    int valueLen = (5 * dimension + dimension * dimension) / 2 + 3;

    if (trainTestData.length == 1) {
      trainTestData[0].initTrain();
      trainTestData[0].setTrain(dataTable.get(0));
      return null;
    }

    double[] rowMatrix = new double[valueLen];
    for (int jj = 0; jj < trainTestData.length; ++jj) {
      Vector vector = dataTable.get(jj);
      rowMatrix[valueLen - 1] += vector.get(valueLen - 1);
      int index = 0;
      for (int i = 0; i < dimension + 1; ++i) {
        rowMatrix[index] = rowMatrix[index] + vector.get(valueLen - 1) / (rowMatrix[valueLen - 1]) * (vector.get(index) - rowMatrix[index]);
        ++index;
      }
      index += dimension;
      for (int i = 0; i < dimension; ++i) {
        for (int j = i; j < dimension; ++j) {
          rowMatrix[index] = (1 - vector.get(valueLen - 1) / rowMatrix[valueLen - 1]) * rowMatrix[index] +
                  vector.get(valueLen - 1) / rowMatrix[valueLen - 1] * vector.get(index) +
                  (1 - vector.get(valueLen - 1) / rowMatrix[valueLen - 1]) *
                          (vector.get(valueLen - 1) / rowMatrix[valueLen - 1]) *
                          (rowMatrix[dimension + 1 + i] - vector.get(dimension + 1 + i)) *
                          (rowMatrix[dimension + 1 + j] - vector.get(dimension + 1 + j));
          ++index;
        }
      }
      rowMatrix[index] = rowMatrix[index] + vector.get(valueLen - 1) / (rowMatrix[valueLen - 1]) * (vector.get(index) - rowMatrix[index]);
      for (int i = 0; i < dimension; ++i) {
        rowMatrix[dimension + 1 + i] = rowMatrix[dimension + 1 + i] +
                vector.get(valueLen - 1) / (rowMatrix[valueLen - 1]) * (vector.get(dimension + 1 + i) - rowMatrix[dimension + 1 + i]);
      }
    }
    TrainTestData wholeData = new TrainTestData();
    Vector whole = new RandomAccessSparseVector(valueLen);
    whole.assign(rowMatrix);
    wholeData.initTrain();
    wholeData.setTrain(whole);

    for (int ii = 0; ii < trainTestData.length; ++ii) {
      rowMatrix = new double[valueLen];
      for (int jj = 0; jj < trainTestData.length; ++jj) {
        if (ii == jj) {
          trainTestData[ii].initTest();
          trainTestData[ii].setTest(dataTable.get(jj));
          continue;
        }
        Vector vector = dataTable.get(jj);
        rowMatrix[valueLen - 1] += vector.get(valueLen - 1);
        int index = 0;
        for (int i = 0; i < dimension + 1; ++i) {
          rowMatrix[index] = rowMatrix[index] + vector.get(valueLen - 1) / (rowMatrix[valueLen - 1]) * (vector.get(index) - rowMatrix[index]);
          ++index;
        }
        index += dimension;
        for (int i = 0; i < dimension; ++i) {
          for (int j = i; j < dimension; ++j) {
            rowMatrix[index] = (1 - vector.get(valueLen - 1) / rowMatrix[valueLen - 1]) * rowMatrix[index] +
                    vector.get(valueLen - 1) / rowMatrix[valueLen - 1] * vector.get(index) +
                    (1 - vector.get(valueLen - 1) / rowMatrix[valueLen - 1]) *
                            (vector.get(valueLen - 1) / rowMatrix[valueLen - 1]) *
                            (rowMatrix[dimension + 1 + i] - vector.get(dimension + 1 + i)) *
                            (rowMatrix[dimension + 1 + j] - vector.get(dimension + 1 + j));
            ++index;
          }
        }
        rowMatrix[index] = rowMatrix[index] + vector.get(valueLen - 1) / (rowMatrix[valueLen - 1]) * (vector.get(index) - rowMatrix[index]);
        for (int i = 0; i < dimension; ++i) {
          rowMatrix[dimension + 1 + i] = rowMatrix[dimension + 1 + i] +
                  vector.get(valueLen - 1) / (rowMatrix[valueLen - 1]) * (vector.get(dimension + 1 + i) - rowMatrix[dimension + 1 + i]);
        }
      }
      Vector train = new RandomAccessSparseVector(valueLen);
      train.assign(rowMatrix);
      trainTestData[ii].initTrain();
      trainTestData[ii].setTrain(train);
    }

    return wholeData;
  }


  /**
   * load all the output from Map-Reduce job into solver
   *
   * @param output the output file path
   * @param config configuration
   * @throws IOException
   */
  public void initSolver(Path output, Configuration config) throws IOException {
    FileSystem fs = FileSystem.get(config);
    FileStatus[] status = fs.listStatus(output);

    List<Vector> dataTable = new ArrayList<Vector>();
    for (int i = 0; i < status.length; ++i) {
      if (PathFilters.partFilter().accept(status[i].getPath())) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, status[i].getPath(), config);
        Text key = (Text) ReflectionUtils.newInstance(reader.getKeyClass(), config);
        VectorWritable value = (VectorWritable) ReflectionUtils.newInstance(reader.getValueClass(), config);
        while (reader.next(key, value)) {
          dimension = Integer.parseInt(key.toString().split("\\.")[0]);
          dataTable.add(value.get());
        }
      }
    }

    trainTestData = new TrainTestData[dataTable.size()];
    wholeData = fillTrainTest(trainTestData, dataTable);
    if(wholeData == null) {
      wholeData = trainTestData[0];
    }

    if(lambdaString.equals("")) {
      setDefaultLambda();
    }
    else {
      setLambda();
    }
  }

  /**
   * Train the model with cross validation.
   */
  public void crossValidate() {
    if(lambda.length == 1) {
      log.warn("lambda sequence has length 1.");
    }
    if(trainTestData.length == 1) {
      log.warn("numOfCV = 1: cross validation is not performed. Please ignore the testing error column in the output");
    }

    trainError = new double[lambda.length];
    testError = new double[lambda.length];
    double[] testStd = new double[lambda.length];
    for(int i = 0;i < trainTestData.length; ++i) {
      TrainTestError[] trainTestErrors = linearSolver(trainTestData[i], lambda, alpha);
      for(int j = 0;j < lambda.length; ++j) {
        trainError[j] += trainTestErrors[j].trainError / trainTestData.length;
        testError[j] += trainTestErrors[j].testError / trainTestData.length;
        testStd[j] += Math.pow(trainTestErrors[j].testError, 2.0);
      }
    }

    int optIndex = 0;
    for(int i = 1;i < lambda.length; ++i) {
      if(testError[optIndex] > testError[i]) {
        optIndex = i;
      }
    }
    optLambda = new double[1];
    double std = Math.sqrt(Math.max(testStd[optIndex] -
            trainTestData.length * Math.pow(testError[optIndex], 2.0), 0) /
            trainTestData.length);
    for(int i = lambda.length - 1; i >= 0; --i) {
      if(testError[i] <= testError[optIndex] + std * PenalizedLinearConstants.RULE_OF_THUMB) {
        optIndex = i;
        break;
      }
    }
    optLambda[0] = lambda[optIndex];
    if(trainTestData.length == 1) {
      optLambda[0] = lambda[0];
    }
    regularizePath(optLambda);
  }
}
