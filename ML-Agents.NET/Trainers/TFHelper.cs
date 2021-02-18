using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Unity3D.Trainers
{
    class tf_helper
    {
        public Tensor polynomial_decay(float learning_rate,
                ResourceVariable global_step,
                float decay_steps,
                float end_learning_rate = 0.0001f,
                float power = 1.0f,
                bool cycle = false,
                string name = null)
            {
                var decayed = new PolynomialDecay(learning_rate,
                    decay_steps,
                    end_learning_rate: end_learning_rate,
                    power: power,
                    cycle: cycle,
                    name: name);

                var decayed_lr = decayed.__call__(global_step);

                return decayed_lr;
            }
    }
}
