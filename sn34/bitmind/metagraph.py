import time
import asyncio
from typing import Callable, List, Tuple
import numpy as np
import bittensor as bt
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from bittensor.utils.weight_utils import process_weights_for_netuid
from substrateinterface import SubstrateInterface

from bitmind.utils import fail_with_none

import threading


def get_miner_uids(
    metagraph: "bt.metagraph", self_uid: int, vpermit_tao_limit: int
) -> List[int]:
    available_uids = []
    for uid in range(int(metagraph.n.item())):
        if uid == self_uid:
            continue

        # Filter non serving axons.
        if not metagraph.axons[uid].is_serving:
            continue
        # Filter validator permit > 1024 stake.
        if metagraph.validator_permit[uid]:
            if metagraph.S[uid] > vpermit_tao_limit:
                continue
        available_uids.append(uid)
        continue
    return available_uids


def create_set_weights(version: int, netuid: int):
    @fail_with_none("Failed setting weights")
    def set_weights(
        wallet: "bt.wallet",
        metagraph: "bt.metagraph",
        subtensor: "bt.subtensor",
        weights: Tuple[List[int], List[float]],
    ):
        uids, raw_weights = weights
        if not len(uids):
            bt.logging.info("No UIDS to score")
            return

        # Set the weights on chain via our subtensor connection.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=np.asarray(uids),
            weights=np.asarray(raw_weights),
            netuid=netuid,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        bt.logging.info("Setting Weights: " + str(processed_weights))
        bt.logging.info("Weight Uids: " + str(processed_weight_uids))
        for _ in range(3):
            result, message = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=processed_weight_uids,  # type: ignore
                weights=processed_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=version,
                max_retries=1,
            )
            if result is True:
                bt.logging.success("set_weights on chain successfully!")
                break
            else:
                bt.logging.error(f"set_weights failed {message}")
            time.sleep(15)
        else:
            bt.logging.error("Sol: set_weights failed after 3 attempts.")

    return set_weights


def create_subscription_handler(substrate, callback: Callable):
    def inner(obj, update_nr, _):
        error = False
        try:
            substrate.get_block(block_number=obj["header"]["number"])
        except Exception as err:
            error = True
            substrate.connect_websocket()
            bt.logging.error(f"Sol: substrate.get_block failed: {err}")
            raise err
        finally:
            if update_nr >= 1:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                retval = loop.run_until_complete(callback(obj["header"]["number"]))
                if not error:
                    return retval

    return inner


def start_subscription(chain_endpoint, callback: Callable):
    while True:
        substrate = SubstrateInterface(
            ss58_format=SS58_FORMAT,
            use_remote_preset=True,
            url=chain_endpoint,
            type_registry=TYPE_REGISTRY,
        )
        try:
            return substrate.subscribe_block_headers(
                create_subscription_handler(substrate, callback)
            )
        except Exception as err:
            bt.logging.error(
                "Sol: create_subscription_handler failed. "
                f"Re-creating substrate: {err}"
            )
            import traceback
            traceback.print_exc()


def run_block_callback_thread(chain_endpoint, callback: Callable):
    try:
        subscription_thread = threading.Thread(
            target=start_subscription, args=[chain_endpoint, callback], daemon=True
        )
        subscription_thread.start()
        bt.logging.info("Block subscription started in background thread.")
        return subscription_thread
    except Exception as e:
        bt.logging.error(f"faaailuuure {callback} - {e}")
