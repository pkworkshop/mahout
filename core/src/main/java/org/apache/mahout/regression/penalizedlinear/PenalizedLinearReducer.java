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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Arrays;

public class PenalizedLinearReducer extends Reducer<Text, VectorWritable, Text, VectorWritable> {

  double[] rowMatrix;
  int dimension;
  int valueLen;
  String emitKey;

  @Override
  protected void reduce(Text key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
    emitKey = key.toString();
    if (rowMatrix == null) {
      dimension = Integer.parseInt(emitKey.split("\\.")[0]);
      valueLen = (5 * dimension + dimension * dimension) / 2 + 3;
      rowMatrix = new double[valueLen];
    }
    Arrays.fill(rowMatrix, 0.0);
    for (VectorWritable value : values) {
      Vector vector = value.get();
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
    VectorWritable emitValue = new VectorWritable();
    Vector vector = new RandomAccessSparseVector(valueLen);
    vector.assign(rowMatrix);
    emitValue.set(vector);
    context.write(new Text(emitKey), emitValue);
  }
}
