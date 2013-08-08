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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;

public class PenalizedLinearTest extends MahoutTestCase {

  private static final double[][] RAW_MAPPER = {{1, 1}, {2, 1}, {1, 2},
      {2, 2}, {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};
  private static final int dimension = 1;

  private static final double[][] RAW_REDUCER = {
      {1.6666666666666667, 1.3333333333333333, 1.3333333333333333, 0.22222222222222224, 2.0, 3.0},
      {15.666666666666668, 3.8333333333333335, 3.8333333333333335, 1.1388888888888888, 15.833333333333334, 6.0}};

  private static final double[] RAW_REDUCER_EXPECT = {11.0, 3.0, 3.0, 2.222222222222222, 11.222222222222221, 9.0};

  private static List<VectorWritable> getPointsWritable(double[][] samples) {
    List<VectorWritable> points = Lists.newArrayList();
    for (double[] fr : samples) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }

  private static VectorWritable getStats(VectorWritable x) {
    Vector vectorx = x.get();
    Vector vector = new RandomAccessSparseVector((5 * dimension + dimension * dimension) / 2 + 3);
    int index = 0;
    for (int i = 1; i < dimension + 1; ++i) {
      vector.set(index++, vectorx.get(i) * vectorx.get(0));
    }
    vector.set(index++, vectorx.get(0));
    for (int i = 1; i < dimension + 1; ++i) {
      vector.set(index++, vectorx.get(i));
    }
    for (int i = 1; i < dimension + 1; ++i) {
      for (int j = i; j < dimension + 1; ++j) {
        vector.set(index++, vectorx.get(i) * vectorx.get(j));
      }
    }
    vector.set(index++, vectorx.get(0) * vectorx.get(0));
    vector.set(index++, 1.0);
    return new VectorWritable(vector);
  }

  private static String getFormatedOutput(VectorWritable vw) {
    String formatedString = "";
    int formatWidth = 8;
    Vector vector = vw.get();
    for (int i = 0; i < vector.size(); ++i) {
      formatedString += String.format("%" + Integer.toString(formatWidth) + ".4f", vector.get(i));
    }
    return formatedString;
  }

  private static String getFormatedOutput(double[] vw) {
    String formatedString = "";
    int formatWidth = 8;
    for (int i = 0; i < vw.length; ++i) {
      formatedString += String.format("%" + Integer.toString(formatWidth) + ".4f", vw[i]);
    }
    return formatedString;
  }

  @BeforeClass
  public static void testSetup() {
  }

  @Test
  public void testPenalizedLinearMapper() throws Exception {
    PenalizedLinearMapper mapper = new PenalizedLinearMapper();
    Configuration conf = getConfiguration();
    DummyRecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, writer);

    List<VectorWritable> points = getPointsWritable(RAW_MAPPER);
    for (VectorWritable vw : points) {
      mapper.map(new Text(), vw, context);
    }
    mapper.cleanup(context);
    assertEquals("Number of map results", 1, writer.getData().size());
    String output = getFormatedOutput(writer.getValue(new Text("1.0")).get(0));
    Vector vectorExpected = new RandomAccessSparseVector((5 * dimension + dimension * dimension) / 2 + 3);
    for (int i = 0; i < points.size(); ++i) {
      VectorWritable vw = getStats(points.get(i));
      for (int j = 0; j < vw.get().size(); ++j) {
        vectorExpected.set(j, vectorExpected.get(j) + vw.get().get(j));
      }
    }
    for (int i = 0; i < vectorExpected.size() - 1; ++i) {
      vectorExpected.set(i, vectorExpected.get(i) / points.size());
    }
    vectorExpected.set(2 * dimension + 1, vectorExpected.get(2 * dimension + 1) - Math.pow(vectorExpected.get(dimension + 1), 2.0));
    String expected = getFormatedOutput(new VectorWritable(vectorExpected));
    assertEquals("Mapper: write-value compare: ", output, expected);
  }

  @Test
  public void testPenalizedLinearReducer() throws Exception {
    PenalizedLinearReducer reducer = new PenalizedLinearReducer();
    Configuration conf = getConfiguration();
    DummyRecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Reducer<Text, VectorWritable, Text, VectorWritable>.Context context = DummyRecordWriter
        .build(reducer, conf, writer, Text.class, VectorWritable.class);

    List<VectorWritable> points = getPointsWritable(RAW_REDUCER);
    reducer.reduce(new Text("1.0"), points, context);
    reducer.reduce(new Text("2.0"), points, context);
    String output = getFormatedOutput(writer.getValue(new Text("1.0")).get(0));
    Vector vector = new RandomAccessSparseVector((5 * dimension + dimension * dimension) / 2 + 3);
    for (int i = 0; i < vector.size(); ++i) {
      vector.set(i, RAW_REDUCER_EXPECT[i]);
    }
    String expect = getFormatedOutput(new VectorWritable(vector));
    assertEquals("Number of reduce results", 2, writer.getData().size());
    assertEquals("Reducer: write-value compare: ", output, expect);
  }

  @Test
  public void testPenalizedLinearDriverAndSolver() throws Exception {
    String[] args = {
        "--input", getTestTempFilePath("testdata/file1").toString(),
        "--output", getTestTempFilePath("testdata/file2").toString(),
        "--alpha", "0.3",
        "--numOfCV", "1",
        "--bias"};
    Configuration conf = new Configuration();

    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, new Path(getTestTempFilePath("testdata/file1").toString()), Text.class, VectorWritable.class);
    try {
      List<VectorWritable> points = getPointsWritable(RAW_MAPPER);
      for (VectorWritable point : points) {
        writer.append(new Text("1"), point);
      }
    } finally {
      writer.close();
    }

    PenalizedLinearDriver pld = new PenalizedLinearDriver();
    pld.run(args);
    PenalizedLinearSolver solver = new PenalizedLinearSolver();
    solver.setAlpha(0.3);
    solver.setIntercept(true);
    solver.setLambdaString("");
    solver.initSolver(getTestTempFilePath("testdata/file2"), conf);
    String lambdaSeq = getFormatedOutput(solver.getLambda());
    String lambdaSeqExpect = "0.0047  0.0050  0.0054  0.0058  0.0062  0.0067  " +
        "0.0071  0.0077  0.0082  0.0088  0.0094  0.0101  0.0108  0.0116  " +
        "0.0125  0.0134  0.0143  0.0154  0.0165  0.0177  0.0190  0.0203  " +
        "0.0218  0.0234  0.0251  0.0269  0.0288  0.0309  0.0331  0.0355  " +
        "0.0381  0.0408  0.0438  0.0470  0.0504  0.0540  0.0579  0.0621  " +
        "0.0666  0.0714  0.0765  0.0821  0.0880  0.0943  0.1012  0.1085  " +
        "0.1163  0.1247  0.1337  0.1434  0.1538  0.1649  0.1768  0.1896  " +
        "0.2033  0.2180  0.2337  0.2506  0.2687  0.2881  0.3089  0.3313  " +
        "0.3552  0.3809  0.4084  0.4379  0.4696  0.5035  0.5399  0.5789  " +
        "0.6207  0.6656  0.7137  0.7653  0.8206  0.8799  0.9435  1.0117  " +
        "1.0848  1.1632  1.2472  1.3374  1.4340  1.5376  1.6488  1.7679  " +
        "1.8957  2.0327  2.1796  2.3371  2.5060  2.6871  2.8813  3.0895  " +
        "3.3128  3.5522  3.8089  4.0841  4.3793  4.6957";
    assertEquals("Default lambda sequence: ", lambdaSeq.trim(), lambdaSeqExpect);
  }
}
