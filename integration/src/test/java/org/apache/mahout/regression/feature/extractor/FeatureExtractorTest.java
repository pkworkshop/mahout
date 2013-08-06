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

package org.apache.mahout.regression.feature.extractor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

public class FeatureExtractorTest extends MahoutTestCase {
  private static final String[] RAW_TXT = {"A B C", "1.0 2.0 3.0", "3.0 4.0 2.0", "2.0 1.0 -1.0"};
  private static final String[] RAW_CSV = {"A,B,C", "1.0,2.0,3.0", "3.0,4.0,2.0", "2.0,1.0,-1.0"};
  private static final double[][] RAW_DATA = {{1.0, 2.0, 3.0, 6.0, 4.0}, {3.0, 4.0, 2.0, 8.0, 16.0}, {2.0, 1.0, -1.0, -1.0, 1.0}};
  private static final String INDEPENDENT = "B,C";
  private static final String INTERACTION = "B:C,B:B";
  private static final String DEPENDENT = "A";
  private static final String SEP_TXT = FeatureExtractUtility.ExtensionToSeparator("txt");
  private static final String SEP_CSV = FeatureExtractUtility.ExtensionToSeparator("csv");


  private static String getFormatedOutput(VectorWritable vw) {
    String formatedString = "";
    int formatWidth = 8;
    Vector vector = vw.get();
    for(int i = 0;i < vector.size(); ++i) {
      formatedString += String.format("%" + Integer.toString(formatWidth) + ".4f", vector.get(i));
    }
    return formatedString;
  }

  private static String getFormatedOutput(double[] vw) {
    String formatedString = "";
    int formatWidth = 8;
    for(int i = 0;i < vw.length; ++i) {
      formatedString += String.format("%" + Integer.toString(formatWidth) + ".4f", vw[i]);
    }
    return formatedString;
  }

  @Test
  public void testExtractorMapperTXT() throws Exception {
    FeatureExtractorMapper mapper = new FeatureExtractorMapper();
    Configuration conf = getConfiguration();
    conf.set("vector.implementation.class.name", "org.apache.mahout.math.RandomAccessSparseVector");
    conf.set(FeatureExtractorKeySet.FEATURE_NAMES, RAW_TXT[0]);
    conf.set(FeatureExtractorKeySet.SELECTED_DEPENDENT, DEPENDENT);
    conf.set(FeatureExtractorKeySet.SELECTED_INDEPENDENT, INDEPENDENT);
    conf.set(FeatureExtractorKeySet.SELECTED_INTERACTION, INTERACTION);
    conf.set(FeatureExtractorKeySet.SEPARATOR, SEP_TXT);
    DummyRecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<LongWritable, Text, Text, VectorWritable>.Context context = DummyRecordWriter
            .build(mapper, conf, writer);
    mapper.setup(context);
    for(int i = 0;i < RAW_TXT.length; ++i) {
      mapper.map(new LongWritable(i), new Text(RAW_TXT[i]), context);
    }
    assertEquals("Number of map results", 1, writer.getData().size());
    for(int i = 0;i < writer.getValue(new Text("5")).size(); ++i) {
      assertEquals("Features: ", getFormatedOutput(writer.getValue(new Text("5")).get(i)), getFormatedOutput(RAW_DATA[i]));
    }
  }

  @Test
  public void testExtractorMapperCSV() throws Exception {
    FeatureExtractorMapper mapper = new FeatureExtractorMapper();
    Configuration conf = getConfiguration();
    conf.set("vector.implementation.class.name", "org.apache.mahout.math.RandomAccessSparseVector");
    conf.set(FeatureExtractorKeySet.FEATURE_NAMES, RAW_CSV[0]);
    conf.set(FeatureExtractorKeySet.SELECTED_DEPENDENT, DEPENDENT);
    conf.set(FeatureExtractorKeySet.SELECTED_INDEPENDENT, INDEPENDENT);
    conf.set(FeatureExtractorKeySet.SELECTED_INTERACTION, INTERACTION);
    conf.set(FeatureExtractorKeySet.SEPARATOR, SEP_CSV);
    DummyRecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<LongWritable, Text, Text, VectorWritable>.Context context = DummyRecordWriter
            .build(mapper, conf, writer);
    mapper.setup(context);
    for(int i = 0;i < RAW_CSV.length; ++i) {
      mapper.map(new LongWritable(i), new Text(RAW_CSV[i]), context);
    }
    assertEquals("Number of map results", 1, writer.getData().size());
    assertEquals("Number of map results", 1, writer.getData().size());
    for(int i = 0;i < writer.getValue(new Text("5")).size(); ++i) {
      assertEquals("Features: ", getFormatedOutput(writer.getValue(new Text("5")).get(i)), getFormatedOutput(RAW_DATA[i]));
    }
  }
}
