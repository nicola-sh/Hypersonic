<?php
use Bitrix\Main\Loader;

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_admin.php");

Loader::includeModule('ipol.aliexpress');

$order = false;

if (isset($_REQUEST['ORDER_ID'])) {
    $order = \Ipol\Aliexpress\DB\OrderTable::findByOrder($_REQUEST['ORDER_ID'], true);
} elseif (isset($_REQUEST['ID'])) {
    $order = \Ipol\Aliexpress\DB\OrderTable::getByPrimary($_REQUEST['ID'], ['select' => ['*']])->fetchObject();
}

if (!$order) {
    CHTTP::SetStatus("404 Not Found");
    @define("ERROR_404","Y");

    ShowError('Заказ не найден');
    
    exit;
}

$request = \Bitrix\Main\Application::getInstance()->getContext()->getRequest();

$form = new \Ipol\Aliexpress\Admin\Form\Order\Edit();
$form->setEntity($order);

$result = $form->processRequest($request);

print $form->render($order->collectValues(), $result);
