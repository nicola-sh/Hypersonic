<?php
use Bitrix\Main\Loader;
use Bitrix\Main\Page\Asset;

Loader::includeModule('ipol.aliexpress');

\CJSCore::Init('ipol_ali_admin_order_edit');

$orderId = $_REQUEST['ORDER_ID'] = $_REQUEST['ID'];

try {
    $order = \Ipol\Aliexpress\DB\OrderTable::findByOrder($orderId, true);


    if ($order) {
        print '
            <script>
                var AliDialogHelper;

                BX.ready(function() {
                    var AliDialogHelper = new AliOrderDetailHelper('. $orderId .');
                });

                function aliMarkSended()
                {
                    AliDialogHelper.process("mark-sended");
                    return false;
                }
            </script>
        ';
    }

} catch (\Exception $e) {

}

?>
