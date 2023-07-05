<?php

use Ipol\AliExpress\Admin\Form\OrderPopup;
use Ipol\AliExpress\Admin\Form;
use Bitrix\Main\Application;

$ALI_ID = false;

//$orderVals = self::$requestVals;
//$status=$orderVals['STATUS'];
$status='NEW';

$request = Application::getInstance()->getContext()->getRequest();
$orderId = $request->get('ID');

$form   = new OrderPopup\Edit();
if (!empty($orderId)) {
    $form->setOrderID($orderId);
}

$entity = new Form\Entity\Options(IPOLH_ALI_MODULE, $form);
$result  = $form->processRequest($request);

?>

<div id="aliForm" style="display: none">
    <?
    print $form->render($entity, $result);
    ?>
</div>

<script>
    var IPOL_ALIEXPRESS = {
        status   : "<?=$status?>",

        form     : 'IPOL_ALI_ORDER_LOGISTIC_POPUP',

        fieldsStep1 : [
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_REFUND_STORE',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_SEND_STORE',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_DIMM_LEN',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_DIMM_WIDTH',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_DIMM_HEIGHT',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_DIMM_WEIGHT'
        ],
        fieldsStep2 : [
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_FIRST_MILE',
        ],
        fieldsStep3 : [
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_RESOURCE_STORE'
        ],
        step1 : true,
        step2 : true,
        step3 : true,

        stockFields : [
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_REFUND_STORE',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_SEND_STORE',
            'IPOL_ALI_ORDER_LOGISTIC_POPUP_ORDER_LOGISTIC_PICKUP_STORE'
        ],

        load: function () {
            if ($('#IPOL_ALI').length <= 0)
                $('.adm-detail-toolbar .adm-detail-toolbar-right').prepend(
                    '<a href="javascript:void(0)" onclick="IPOL_ALIEXPRESS.showWindow()" class="adm-btn" id="IPOL_ALI">Aliexpress доставка</a>');
        },

        showStockInfo: function (field, notCheckSolutions) {
            var text = $('#'+field).find('option:selected').data('info'),
                html = '<tr><td colspan="2"><div id="'+field+'_info" class="adm-info-message" style="width: 100%; box-sizing: border-box;">'+text+'</div></td></tr>';

            if ($('#'+field+'_info').length > 0)
                $('#'+field+'_info').text(text);
            else
                $('#'+field).parent('td').parent('tr').before(html);

            if (typeof notCheckSolutions !== 'undefined' && !notCheckSolutions)
                IPOL_ALIEXPRESS.getSolutions();
        },

        checkFieldSet: function(fieldSet) {
            var fieldsCheck = true;
            for (var i in fieldSet) {
                if ($('#'+fieldSet[i]).val().length <= 0) {
                    fieldsCheck = false;
                }
            }
            return fieldsCheck;
        },

        checkForm: function() {
            // $('#IPOL_ALIEXPRESS_get_solutions').prop('disabled', true);
            // $('#IPOL_ALIEXPRESS_set_first_mile').prop('disabled', true);
            $('#IPOL_ALIEXPRESS_create_logistic_order').prop('disabled', true);

            IPOL_ALIEXPRESS.step1 = IPOL_ALIEXPRESS.checkFieldSet(IPOL_ALIEXPRESS.fieldsStep1);
            IPOL_ALIEXPRESS.step2 = IPOL_ALIEXPRESS.checkFieldSet(IPOL_ALIEXPRESS.fieldsStep2);
            IPOL_ALIEXPRESS.step3 = IPOL_ALIEXPRESS.checkFieldSet(IPOL_ALIEXPRESS.fieldsStep3);

            // if (IPOL_ALIEXPRESS.step1) $('#IPOL_ALIEXPRESS_get_solutions').prop('disabled', false);
            // if (IPOL_ALIEXPRESS.step2 && IPOL_ALIEXPRESS.step1) $('#IPOL_ALIEXPRESS_set_first_mile').prop('disabled', false);
            if (IPOL_ALIEXPRESS.step3 && IPOL_ALIEXPRESS.step2 && IPOL_ALIEXPRESS.step1) $('#IPOL_ALIEXPRESS_create_logistic_order').prop('disabled', false);
        },

        wnd: false,

        showWindow: function() {
            var savButStat='';
            if(IPOL_ALIEXPRESS.status!='ERROR' && IPOL_ALIEXPRESS.status!='NEW')
                savButStat='style="display:none"';
            var delButStat='';
            if(IPOL_ALIEXPRESS.status !='OK' && IPOL_ALIEXPRESS.status !='ERROR' && IPOL_ALIEXPRESS.status !='DELETED' )
                delButStat='style="display:none"';
            var prntButStat='style="display:none"';
            if(IPOL_ALIEXPRESS.status =='OK')
                prntButStat='';

            if(!IPOL_ALIEXPRESS.wnd){
                var html=$('#aliForm').html();
                $('#aliForm').html('');
                IPOL_ALIEXPRESS.wnd = new BX.CDialog({
                    title: "<?=GetMessage('IPOL_ALIEXPRESS_JSC_SOD_WNDTITLE')?>",
                    content: html,
                    icon: 'head-block',
                    resizable: true,
                    draggable: true,
                    height: '500',
                    width: '805',
                    buttons: [
                        //'<input disabled type=\"button\" id=\"IPOL_ALIEXPRESS_get_solutions\" value=\"<?//=GetMessage('IPOL_ALIEXPRESS_JSC_GET_SOLUTIONS')?>//\" onclick=\"IPOL_ALIEXPRESS.getSolutions();\">',
                        //'<input disabled type=\"button\" id=\"IPOL_ALIEXPRESS_set_first_mile\" value=\"<?//=GetMessage('IPOL_ALIEXPRESS_JSC_SET_FIRST_MILE')?>//\" onclick=\"IPOL_ALIEXPRESS.setFirstMile();\">',
                        '<input disabled type=\"button\" id=\"IPOL_ALIEXPRESS_create_logistic_order\" value=\"<?=GetMessage('IPOL_ALIEXPRESS_JSC_LOG_ORDER_CREATE')?>\" onclick=\"IPOL_ALIEXPRESS.createOrder();\">',

                        <?if($ALI_ID){?>'<a href="<?=\Ipolh\SDEK\SDEK\Tools::getTrackLink($ALI_ID)?>" target="_blank"><?=GetMessage('IPOL_ALIEXPRESS_JSC_SOD_FOLLOW')?></a>'<?}?> // follow
                    ]
                });
                $('#IPOL_ALIEXPRESS_courierTimeBeg').mask("29:59");
                $('#IPOL_ALIEXPRESS_courierTimeEnd').mask("29:59");
                // $('#IPOL_ALIEXPRESS_courierPhone').mask("99999999999");

                $( "#IPOL_ALIEXPRESS_cSSelector" ).autocomplete({
                    source: IPOL_ALIEXPRESS.senderCities,
                    select: function(ev,ui){IPOL_ALIEXPRESS.courier.changeCity(2,ui);}
                });
            }
            IPOL_ALIEXPRESS.wnd.Show();
        },

        callAction: function (data) {
            $.ajax({
                data: data,
                url: '/bitrix/js/ipol.aliexpress/handler.php',
                type: 'post',
                dataType: 'json',
                success: function(data) {
                    if (typeof data['form'] !== 'undefined')
                        $('#'+IPOL_ALIEXPRESS.form).replaceWith(data.form);

                    for (var i in IPOL_ALIEXPRESS.stockFields) {
                        IPOL_ALIEXPRESS.showStockInfo(IPOL_ALIEXPRESS.stockFields[i], true);
                    }
                }
            });
        },

        getSolutions: function () {
            var data = $('#'+IPOL_ALIEXPRESS.form).serialize()+'&action=getSolutions&orderId=<?=$orderId?>';
            IPOL_ALIEXPRESS.callAction(data)
        },

        getResources: function () {
            var data = $('#'+IPOL_ALIEXPRESS.form).serialize()+'&action=getResources&orderId=<?=$orderId?>';
            IPOL_ALIEXPRESS.callAction(data)
        },

        setFirstMile: function () {
            var data = $('#'+IPOL_ALIEXPRESS.form).serialize()+'&action=setFirstMile';
            IPOL_ALIEXPRESS.callAction(data)
        },

        createOrder: function() {
            var data = $('#'+IPOL_ALIEXPRESS.form).serialize()+'&action=createOrder&orderId=<?=$orderId?>';
            IPOL_ALIEXPRESS.callAction(data)
        },
    };

    $(document).ready(IPOL_ALIEXPRESS.load);
</script>