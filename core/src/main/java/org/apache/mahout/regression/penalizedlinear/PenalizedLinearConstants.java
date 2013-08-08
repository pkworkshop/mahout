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

public class PenalizedLinearConstants {
  public static final double CONVERGENCE_THRESHOLD = 1E-6;
  public static final double MINIMUM_VARIANCE = 1E-4;
  public static final int MAXIMUM_PASS = 5000;
  public static final double ALPHA_THRESHOLD = 1E-4;
  public static final int DEFAULT_NUM_LAMBDA = 100;
  public static final double LAMBDA_RATIO = 1E-3;
  public static final double LAMBDA_MULTIPLIER = 1.05;
  public static final double RULE_OF_THUMB = 0.5;

  private PenalizedLinearConstants() {
    // Empty for ensuring that this class cannot be instantiated.
  }
}
