class EfficientPredictionResults:
    def __init__(self):
        self.results = {}
        self.results_min = {}

        self.qrels = {}

    def _is_update_target(self, qid, pred):
        min_val = self._get_min_val(qid)
        return pred > min_val

    # ================= Min val control ================= #
    def _init_min(self, qid):
        if qid not in self.results_min:
            self.results_min[qid] = {}

    def _get_min_did(self, qid):
        return self.results_min[qid]

    def _get_min_val(self, qid):
        return self.results[qid][self._get_min_did(qid)]
    def _update_min(self, qid, pid, pred):
        self._init_min(qid)
        self.results_min[qid] = pid

    # ================= Res val control ================= #
    def _init_res(self, qid):
        if qid not in self.results:
            self.results[qid] = {}

    def _del_res(self, qid, pid):
        del self.results[qid][pid]

    def _update_res(self, qid, pid, pred):
        self._init_res(qid)
        self.results[qid][pid] = pred

    # ================= Qrels val control ================= #
    def _init_qrels(self, qid):
        if qid not in self.qrels:
            self.qrels[qid] = {}
    def _update_qrels(self, qid, pid, label):
        if type(label) is int and label <= 0:
            return
        if type(label) is bool and label is False:
            return

        self._init_qrels(qid)
        self.qrels[qid][pid] = 1

    def update(self, qid, pid, pred, label):
        self._update_qrels(qid, pid, label)

        if self._is_update_target(qid, pred):
            self._update_res(qid, pid, pred)
            self._del_res(qid, self._get_min_did(qid))
            self._update_min(qid, pid, pred)




