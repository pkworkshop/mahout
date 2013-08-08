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

import com.google.common.collect.Maps;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

/**
 * In each line, the first element of VectorWritable is response; the rest are predictors.
 */
public class PenalizedLinearMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private static final Logger log = LoggerFactory.getLogger(PenalizedLinearMapper.class);
  int numberOfCV = 0;
  Random rand = new Random();
  Map<String, double[]> rowMatrix;
  int dimension;
  int valueLen;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context) throws IOException, InterruptedException {
    if (rowMatrix == null) {
      rowMatrix = Maps.newHashMap();
      numberOfCV = context.getConfiguration().getInt(PenalizedLinearKeySet.NUM_CV, 1);
      dimension = value.get().size() - 1;
      if (dimension <= 0) {
        String err = "Fatal: Input file has dimension <= 0!";
        log.error(err);
        throw (new IOException(err));
      }
      valueLen = (5 * dimension + dimension * dimension) / 2 + 3;
      for (int i = 0; i < numberOfCV; ++i) {
        double[] vector = new double[valueLen];
        rowMatrix.put(String.valueOf(dimension) + "." + String.valueOf(i), vector);
      }
    }

    Vector sample = value.get();
    String emitKey = String.valueOf(dimension) + "." + Integer.toString(rand.nextInt(numberOfCV));
    double[] emitValue = rowMatrix.get(emitKey);

    ++emitValue[valueLen - 1];

    double y = sample.get(0);
    int index = 0;
    for (int jj = 1; jj < dimension + 1; ++jj) {
      emitValue[index] = emitValue[index] +
          (sample.get(jj) * y - emitValue[index]) / emitValue[valueLen - 1];
      ++index;
    }

    emitValue[index] = emitValue[index] +
        (y - emitValue[index]) / emitValue[valueLen - 1];
    ++index;
    index += dimension;
    for (int jj = 1; jj < dimension + 1; ++jj) {
      for (int kk = jj; kk < dimension + 1; ++kk) {
        emitValue[index] = emitValue[index] * (1 - 1.0 / emitValue[valueLen - 1]) +
            (emitValue[dimension + jj] - sample.get(jj)) *
                (emitValue[dimension + kk] - sample.get(kk)) *
                (1 - 1.0 / emitValue[valueLen - 1]) / emitValue[valueLen - 1];
        ++index;
      }
    }
    emitValue[index] = emitValue[index] + (y * y - emitValue[index]) / emitValue[valueLen - 1];
    index = dimension + 1;
    for (int jj = 1; jj < dimension + 1; ++jj) {
      emitValue[index] = emitValue[index] + (sample.get(jj) - emitValue[index]) / emitValue[valueLen - 1];
      ++index;
    }
    rowMatrix.put(emitKey, emitValue);
  }

  @Override
  public void cleanup(Context context) throws IOException, InterruptedException {
    VectorWritable emitValue = new VectorWritable();
    Vector vector = new RandomAccessSparseVector(valueLen);
    for (int i = 0; i < numberOfCV; ++i) {
      String key = String.valueOf(dimension) + "." + String.valueOf(i);
      vector.assign(rowMatrix.get(key));
      emitValue.set(vector);
      context.write(new Text(key), emitValue);
    }
  }
}
