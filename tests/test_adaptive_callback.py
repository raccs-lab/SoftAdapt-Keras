import numpy as np

from keras import layers, losses, optimizers, ops, random
from keras.src import testing
from keras.src.models.sequential import Sequential
from softadapt import AdaptiveLossCallback, algorithms


class AdaptiveLossCallbackTest(testing.TestCase):
    # @pytest.mark.requires_trainable_backend
    def test_adaptive_callback(self):
        """Test standard AdaptiveLossCallback functionalities with training."""
        batch_size = 4
        # Create a small test model
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=batch_size), layers.Dense(1)]
        )
        # Use minimal set of losses
        loss_list = [losses.MeanSquaredError(), losses.MeanAbsoluteError()]
        # Compile with pre-defined weights
        model.compile(
            optimizer=optimizers.SGD(),
            loss=loss_list,
            loss_weights=[0.1, 0.9],
        )
        # Generate random data
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        # Initialize new adaptive callback
        callback_obj = AdaptiveLossCallback(
            components=[lss.name for lss in loss_list],
            frequency="epoch",
            beta=0.1,
            algorithm="base",
        )
        # Run a few fit iterations
        model.fit(
            x,
            y,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[callback_obj],
            epochs=5,
            verbose=0,
        )

        # Ensure the loss weights have been updated
        assert np.allclose(ops.convert_to_numpy(callback_obj.weights), [0.5, 0.5])

    def test_callback_initialization(self):
        """Test for callback initialization"""
        callback_obj = AdaptiveLossCallback(
            components=["component 1", "component 2"],
            frequency="epoch",
            beta=0.1,
            algorithm="base",
        )
        assert isinstance(callback_obj.algorithm, algorithms.SoftAdapt)
        assert callback_obj.frequency == "epoch"
        assert len(callback_obj.components_history) == len(callback_obj.order)

    def test_on_epoch_end_updates_weights(self):
        """Test that weights are updated after two epochs"""
        model = Sequential([layers.Input(shape=(2,), batch_size=4), layers.Dense(1)])
        loss_list = [losses.MeanSquaredError(), losses.MeanAbsoluteError()]
        model.compile(
            optimizer=optimizers.SGD(),
            loss=loss_list,
            loss_weights=[0.9, 0.1],
        )

        callback_obj = AdaptiveLossCallback(
            components=[lss.name for lss in loss_list],
            frequency="epoch",
            beta=0.1,
            algorithm="base",
        )

        callback_obj.set_model(model)

        model.compute_loss(y_pred=random.uniform((4, 2)), y=random.uniform((4, 1)))

        logs = {"mean_squared_error_loss": 1.0, "mean_absolute_error_loss": 0.3}
        callback_obj.on_epoch_end(epoch=0, logs=logs)
        callback_obj.on_epoch_end(epoch=1, logs=logs)

        # Check if the component history is updated
        assert np.allclose(
            ops.convert_to_numpy(callback_obj.components_history[0]), [1.0]
        )
        assert np.allclose(
            ops.convert_to_numpy(callback_obj.components_history[1]), [0.3]
        )

        # Check if the weights have been updated
        assert np.allclose(ops.convert_to_numpy(callback_obj.weights), [0.5, 0.5])

    def test_on_epoch_end_clears_history(self):
        class MockModel:
            @property
            def loss_weights(self):
                return [0, 0]

            @loss_weights.setter
            def loss_weights(self, value):
                pass

        model = MockModel()
        callback_obj = AdaptiveLossCallback(
            components=["loss1", "loss2"],
            frequency="epoch",
            beta=0.1,
            algorithm="base",
        )

        callback_obj.set_model(model)
        logs = {"loss1_loss": 0.2, "loss2_loss": 0.3}
        callback_obj.on_epoch_end(epoch=0, logs=logs)

        # Check if the component history is updated
        assert callback_obj.components_history[0] == [0.2]
        assert callback_obj.components_history[1] == [0.3]

        logs = {"loss1_loss": 0.5, "loss2_loss": 0.2}
        callback_obj.on_epoch_end(epoch=1, logs=logs)

        # Check if the history has been cleared except for one
        assert callback_obj.components_history[0] == [0.5]
        assert callback_obj.components_history[1] == [0.2]
