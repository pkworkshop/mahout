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

import com.google.common.collect.Lists;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.List;

/**
 * This class converts white-spaced TEXT files or CSV files into
 * Mahout sequence files of VectorWritable suitable for input to the clustering jobs in
 * particular, and any Mahout job requiring this input in general.
 */

public class FeatureExtractorMapper extends Mapper<LongWritable, Text, Text, VectorWritable> {

  List<Pair<Integer, Integer>> interactionPairList;
  int[] independentID;
  int dependentID;
  String pattern;
  private Constructor<?> constructor;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    String vectorImplClassName = context.getConfiguration().get("vector.implementation.class.name");
    try {
      Class<? extends Vector> outputClass = context.getConfiguration().getClassByName(vectorImplClassName).asSubclass(Vector.class);
      constructor = outputClass.getConstructor(int.class);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }

    String featureNames = context.getConfiguration().get(FeatureExtractorKeySet.FEATURE_NAMES);
    String selectedIndependent = context.getConfiguration().get(FeatureExtractorKeySet.SELECTED_INDEPENDENT);
    String selectedInteraction = context.getConfiguration().get(FeatureExtractorKeySet.SELECTED_INTERACTION);
    String selectedDependent = context.getConfiguration().get(FeatureExtractorKeySet.SELECTED_DEPENDENT);
    String separator = context.getConfiguration().get(FeatureExtractorKeySet.SEPARATOR);
    String[] features;
    pattern = FeatureExtractUtility.SeparatorToPattern(separator);
    features = featureNames.trim().split(pattern);
    HashMap<String, Integer> featureMap = new HashMap<String, Integer>();
    for (int i = 0; i < features.length; ++i) {
      featureMap.put(features[i], new Integer(i));
    }
    if (selectedIndependent != null && !selectedIndependent.equals("")) {
      String[] independents = selectedIndependent.split(",");
      independentID = new int[independents.length];
      for (int i = 0; i < independents.length; ++i) {
        independentID[i] = featureMap.get(independents[i]);
      }
    }

    if (selectedInteraction != null && !selectedInteraction.equals("")) {
      String[] interactions = selectedInteraction.split(",");
      interactionPairList = Lists.newArrayList();
      for (int i = 0; i < interactions.length; ++i) {
        Integer first = new Integer(featureMap.get(interactions[i].split(":")[0]));
        Integer second = new Integer(featureMap.get(interactions[i].split(":")[1]));
        interactionPairList.add(new Pair<Integer, Integer>(first, second));
      }
    }
    dependentID = new Integer(featureMap.get(selectedDependent));
  }

  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    if (key.get() != 0L) {
      String[] numberString = value.toString().trim().split(pattern);
      try {
        int size = 1;
        if (interactionPairList != null) {
          size += interactionPairList.size();
        }
        if (independentID != null) {
          size += independentID.length;
        }
        Vector result = (Vector) constructor.newInstance(size);
        result.set(0, Double.valueOf(numberString[dependentID]));
        int index = 1;
        if (independentID != null) {
          for (int i = 0; i < independentID.length; ++i) {
            result.set(index++, Double.valueOf(numberString[independentID[i]]));
          }
        }
        if (interactionPairList != null) {
          for (int i = 0; i < interactionPairList.size(); ++i) {
            Pair<Integer, Integer> pair = interactionPairList.get(i);
            result.set(index++, Double.valueOf(numberString[pair.getFirst()]) * Double.valueOf(numberString[pair.getSecond()]));
          }
        }
        VectorWritable vectorWritable = new VectorWritable(result);
        context.write(new Text(String.valueOf(index)), vectorWritable);
      } catch (InstantiationException e) {
        throw new IllegalStateException(e);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      } catch (InvocationTargetException e) {
        throw new IllegalStateException(e);
      }
    }
  }
}
